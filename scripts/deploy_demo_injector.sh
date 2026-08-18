#!/usr/bin/env bash
# Package and deploy demo_injector.py as the memory-demo-injector Lambda, and
# put it on a 30-minute EventBridge schedule.
#
#   ./scripts/deploy_demo_injector.sh
#
# This is the only scheduled component that WRITES.  It re-injects the two
# retrieval failures every 30 minutes so the checks' 60-minute windows are
# populated for the whole judging period, and prunes demo rows older than 24h
# so 48 runs a day cannot grow the cluster without bound.
#
# It is deliberately a separate function, role and schedule from
# memory-health-checks: the checker reads, this writes, and the two should not
# share an identity.
#
# Requires: aws CLI (configured), python3.12, and DATABASE_URL in .env.
# Re-running updates the function in place.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
FUNC="${INJECTOR_FUNCTION_NAME:-memory-demo-injector}"
ROLE_NAME="${FUNC}-lambda-role"
SCHED_ROLE_NAME="${FUNC}-scheduler-role"
SCHEDULE_NAME="${FUNC}-every-30min"
DEPLOYER_POLICY="MemoryDemoInjectorScheduler"
EMBED_MODEL="amazon.titan-embed-text-v2:0"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
CALLER_ARN="$(aws sts get-caller-identity --query Arn --output text)"
echo "account=$ACCOUNT region=$REGION function=$FUNC"

# --- 1. build the package with Linux wheels, not the host's -------------------
echo "building deployment package..."
# typing_extensions is a hard runtime import for psycopg on any Python < 3.13
# (psycopg/_compat.py), but pip does not resolve it as a dependency under
# --python-version 3.12, so it has to be named explicitly. Omitting it produces
# a package that imports cleanly on the host and dies in Lambda with
# "No module named 'typing_extensions'".
python3 -m pip install --quiet \
  --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.12 \
  --target "$BUILD/pkg" "psycopg[binary]==3.3.4" "python-dotenv==1.2.2" \
  "typing-extensions>=4.6"
cp "$ROOT/demo_injector.py" "$ROOT/demo_harness.py" \
   "$ROOT/checks.py" "$ROOT/memory.py" "$BUILD/pkg/"

# psycopg's bundled libpq cannot resolve the container trust store, so ship the
# CockroachDB CA explicitly and point sslrootcert at it (see README).
if [ -f "$ROOT/certs/cockroachdb-root.crt" ]; then
  cp "$ROOT/certs/cockroachdb-root.crt" "$BUILD/pkg/root.crt"
elif [ -f "$HOME/.postgresql/root.crt" ]; then
  cp "$HOME/.postgresql/root.crt" "$BUILD/pkg/root.crt"
else
  echo "ERROR: no CockroachDB CA found (certs/cockroachdb-root.crt or ~/.postgresql/root.crt)."
  exit 1
fi
(cd "$BUILD/pkg" && zip -qr "$BUILD/lambda.zip" . -x "*.dist-info/*")
echo "package: $(du -h "$BUILD/lambda.zip" | cut -f1)"

# --- 2. IAM execution role ----------------------------------------------------
# Its own role, not the checker's: this identity can write, and nothing that
# only needs to read should inherit that.
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "creating $ROLE_NAME"
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --query 'Role.Arn' --output text
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "waiting for IAM propagation..."; sleep 12
fi

# The injections run real recall() calls, which embed each query with Titan.
# Scoped to that one model -- this function has no business calling any other.
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name InvokeTitanEmbed \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"bedrock:InvokeModel\",\"Resource\":\"arn:aws:bedrock:${REGION}::foundation-model/${EMBED_MODEL}\"}]}"
ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}"

# --- 3. environment (never echoed) -------------------------------------------
ENV_JSON="$BUILD/env.json"
python3 - "$ROOT" "$ENV_JSON" <<'PY'
import json, pathlib, sys, urllib.parse
root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
url = None
for line in (root / ".env").read_text().splitlines():
    if line.strip().startswith("DATABASE_URL="):
        url = line.split("=", 1)[1].strip()
if not url:
    raise SystemExit("DATABASE_URL not found in .env")
if len(url) >= 2 and url[0] == url[-1] and url[0] in ("'", '"'):
    url = url[1:-1]
if not url.startswith(("postgres://", "postgresql://")):
    raise SystemExit("DATABASE_URL does not look like a postgres connection string")
p = urllib.parse.urlparse(url)
q = urllib.parse.parse_qs(p.query)
q["sslrootcert"] = ["/var/task/root.crt"]     # the CA we shipped in the zip
q.setdefault("sslmode", ["verify-full"])       # never downgrade TLS
url = urllib.parse.urlunparse(p._replace(query=urllib.parse.urlencode(q, doseq=True)))
out.write_text(json.dumps({"Variables": {"DATABASE_URL": url}}))
print("env prepared (value not printed)")
PY
chmod 600 "$ENV_JSON"

# --- 4. create or update the function ------------------------------------------
if aws lambda get-function --function-name "$FUNC" --region "$REGION" >/dev/null 2>&1; then
  echo "updating $FUNC"
  aws lambda update-function-code --function-name "$FUNC" --region "$REGION" \
    --zip-file "fileb://$BUILD/lambda.zip" --query 'LastUpdateStatus' --output text
  aws lambda wait function-updated-v2 --function-name "$FUNC" --region "$REGION"
  aws lambda update-function-configuration --function-name "$FUNC" --region "$REGION" \
    --environment "file://$ENV_JSON" --timeout 180 --memory-size 512 \
    --query 'LastUpdateStatus' --output text
else
  echo "creating $FUNC"
  aws lambda create-function --function-name "$FUNC" --region "$REGION" \
    --runtime python3.12 --architectures x86_64 --role "$ROLE_ARN" \
    --handler demo_injector.lambda_handler --zip-file "fileb://$BUILD/lambda.zip" \
    --memory-size 512 --timeout 180 --environment "file://$ENV_JSON" \
    --query 'FunctionArn' --output text
fi
aws lambda wait function-updated-v2 --function-name "$FUNC" --region "$REGION"
rm -f "$ENV_JSON"
FUNC_ARN="arn:aws:lambda:${REGION}:${ACCOUNT}:function:${FUNC}"

# --- 5. scheduler role ---------------------------------------------------------
if ! aws iam get-role --role-name "$SCHED_ROLE_NAME" >/dev/null 2>&1; then
  echo "creating $SCHED_ROLE_NAME"
  aws iam create-role --role-name "$SCHED_ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --query 'Role.Arn' --output text
  echo "waiting for IAM propagation..."; sleep 12
fi
aws iam put-role-policy --role-name "$SCHED_ROLE_NAME" --policy-name InvokeInjector \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"lambda:InvokeFunction\",\"Resource\":\"${FUNC_ARN}\"}]}"
SCHED_ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${SCHED_ROLE_NAME}"

# --- 6. the deploying principal's scheduler grant ------------------------------
# AWSLambda_FullAccess carries no EventBridge Scheduler permissions, so the
# deployer needs them added explicitly.  Same shape as
# MemoryHealthChecksScheduler: scheduler verbs restricted to this schedule's
# name prefix, and PassRole restricted to this one role AND to the scheduler
# service, so the grant cannot be used to hand any other role to anything else.
if [[ "$CALLER_ARN" == *":user/"* ]]; then
  USER_NAME="${CALLER_ARN##*/}"
  echo "granting $DEPLOYER_POLICY to $USER_NAME"
  aws iam put-user-policy --user-name "$USER_NAME" --policy-name "$DEPLOYER_POLICY" \
    --policy-document "{
      \"Version\": \"2012-10-17\",
      \"Statement\": [
        {
          \"Sid\": \"ManageMemoryDemoInjectorSchedules\",
          \"Effect\": \"Allow\",
          \"Action\": [
            \"scheduler:CreateSchedule\",
            \"scheduler:GetSchedule\",
            \"scheduler:UpdateSchedule\",
            \"scheduler:DeleteSchedule\"
          ],
          \"Resource\": \"arn:aws:scheduler:${REGION}:${ACCOUNT}:schedule/default/${FUNC}-*\"
        },
        {
          \"Sid\": \"PassSchedulerRoleToEventBridgeSchedulerOnly\",
          \"Effect\": \"Allow\",
          \"Action\": \"iam:PassRole\",
          \"Resource\": \"${SCHED_ROLE_ARN}\",
          \"Condition\": {
            \"StringEquals\": { \"iam:PassedToService\": \"scheduler.amazonaws.com\" }
          }
        }
      ]
    }"
  echo "waiting for IAM propagation..."; sleep 12
else
  echo "caller is not an IAM user ($CALLER_ARN) -- skipping the deployer grant."
  echo "Ensure the principal can scheduler:CreateSchedule on ${FUNC}-* and PassRole ${SCHED_ROLE_NAME}."
fi

# --- 7. the schedule -----------------------------------------------------------
if aws scheduler get-schedule --name "$SCHEDULE_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "updating schedule $SCHEDULE_NAME"
  aws scheduler update-schedule --name "$SCHEDULE_NAME" --region "$REGION" \
    --schedule-expression "rate(30 minutes)" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --state ENABLED \
    --target "{\"Arn\":\"${FUNC_ARN}\",\"RoleArn\":\"${SCHED_ROLE_ARN}\"}" \
    --query 'ScheduleArn' --output text
else
  echo "creating schedule $SCHEDULE_NAME"
  aws scheduler create-schedule --name "$SCHEDULE_NAME" --region "$REGION" \
    --schedule-expression "rate(30 minutes)" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --state ENABLED \
    --target "{\"Arn\":\"${FUNC_ARN}\",\"RoleArn\":\"${SCHED_ROLE_ARN}\"}" \
    --query 'ScheduleArn' --output text
fi

echo
echo "deployed. invoke once with:"
echo "  aws lambda invoke --function-name $FUNC --region $REGION /dev/stdout"
echo "pause the demo refresh with:"
echo "  aws scheduler update-schedule --name $SCHEDULE_NAME --region $REGION \\"
echo "    --schedule-expression 'rate(30 minutes)' --flexible-time-window '{\"Mode\":\"OFF\"}' \\"
echo "    --target '{\"Arn\":\"$FUNC_ARN\",\"RoleArn\":\"$SCHED_ROLE_ARN\"}' --state DISABLED"

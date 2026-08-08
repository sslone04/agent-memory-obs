#!/usr/bin/env bash
# Package and deploy checks.py as the memory-health-checks Lambda, and put it
# on a 5-minute EventBridge schedule.
#
#   ./scripts/deploy_lambda.sh
#
# Requires: aws CLI (configured), python3.12, and DATABASE_URL in .env.
# Re-running updates the function in place.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
FUNC="${LAMBDA_FUNCTION_NAME:-memory-health-checks}"
ROLE_NAME="${FUNC}-lambda-role"
SCHED_ROLE_NAME="${FUNC}-scheduler-role"
SCHEDULE_NAME="${FUNC}-every-5min"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
echo "account=$ACCOUNT region=$REGION function=$FUNC"

# --- 1. build the package with Linux wheels, not the host's -------------------
echo "building deployment package..."
python3 -m pip install --quiet \
  --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.12 \
  --target "$BUILD/pkg" "psycopg[binary]==3.3.4" "python-dotenv==1.2.2"
cp "$ROOT/checks.py" "$ROOT/memory.py" "$BUILD/pkg/"

# psycopg's bundled libpq cannot resolve the container trust store, so ship the
# CockroachDB CA explicitly and point sslrootcert at it (see README).
if [ -f "$HOME/.postgresql/root.crt" ]; then
  cp "$HOME/.postgresql/root.crt" "$BUILD/pkg/root.crt"
else
  echo "ERROR: ~/.postgresql/root.crt not found."
  echo "Download it from the CockroachDB Cloud console (Connect -> CA cert) first."
  exit 1
fi
(cd "$BUILD/pkg" && zip -qr "$BUILD/lambda.zip" . -x "*.dist-info/*")
echo "package: $(du -h "$BUILD/lambda.zip" | cut -f1)"

# --- 2. IAM execution role ----------------------------------------------------
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "creating $ROLE_NAME"
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --query 'Role.Arn' --output text
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "waiting for IAM propagation..."; sleep 12
fi
ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}"

# --- 3. environment (never echoed) -------------------------------------------
ENV_JSON="$BUILD/env.json"
python3 - "$ROOT" "$ENV_JSON" <<'PY'
import json, os, pathlib, sys, urllib.parse
root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
url = None
for line in (root / ".env").read_text().splitlines():
    if line.strip().startswith("DATABASE_URL="):
        url = line.split("=", 1)[1].strip()
if not url:
    raise SystemExit("DATABASE_URL not found in .env")
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
    --environment "file://$ENV_JSON" --query 'LastUpdateStatus' --output text
else
  echo "creating $FUNC"
  aws lambda create-function --function-name "$FUNC" --region "$REGION" \
    --runtime python3.12 --architectures x86_64 --role "$ROLE_ARN" \
    --handler checks.lambda_handler --zip-file "fileb://$BUILD/lambda.zip" \
    --memory-size 512 --timeout 60 --environment "file://$ENV_JSON" \
    --query 'FunctionArn' --output text
fi
aws lambda wait function-updated-v2 --function-name "$FUNC" --region "$REGION"
rm -f "$ENV_JSON"
FUNC_ARN="arn:aws:lambda:${REGION}:${ACCOUNT}:function:${FUNC}"

# --- 5. EventBridge Scheduler --------------------------------------------------
if ! aws iam get-role --role-name "$SCHED_ROLE_NAME" >/dev/null 2>&1; then
  echo "creating $SCHED_ROLE_NAME"
  aws iam create-role --role-name "$SCHED_ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --query 'Role.Arn' --output text
  aws iam put-role-policy --role-name "$SCHED_ROLE_NAME" --policy-name InvokeChecks \
    --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"lambda:InvokeFunction\",\"Resource\":\"${FUNC_ARN}\"}]}"
  echo "waiting for IAM propagation..."; sleep 12
fi
SCHED_ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${SCHED_ROLE_NAME}"

if aws scheduler get-schedule --name "$SCHEDULE_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "schedule $SCHEDULE_NAME already exists"
else
  aws scheduler create-schedule --name "$SCHEDULE_NAME" --region "$REGION" \
    --schedule-expression "rate(5 minutes)" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --target "{\"Arn\":\"${FUNC_ARN}\",\"RoleArn\":\"${SCHED_ROLE_ARN}\"}" \
    --query 'ScheduleArn' --output text
fi

echo
echo "deployed. invoke once with:"
echo "  aws lambda invoke --function-name $FUNC --region $REGION /dev/stdout"

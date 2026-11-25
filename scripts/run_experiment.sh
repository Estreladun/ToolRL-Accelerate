#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-mac_smoke}"
case "$PROFILE" in
  cloud_smoke|cloud3b) exec bash scripts/run_cloud.sh "$PROFILE" "${2:-summary}";;
  mac_smoke|mac128) ;;
  *) echo 'Profile must be mac_smoke, mac128, cloud_smoke or cloud3b'; exit 2;;
esac
while IFS='=' read -r key value; do
  [[ "$key" =~ ^[A-Z_]+$ ]] || continue
  if [[ -z "${!key+x}" ]]; then export "$key=$value"; fi
done < "configs/$PROFILE.env"
bash scripts/run_mac.sh "${2:-summary}"

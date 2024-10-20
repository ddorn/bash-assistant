#!/bin/bash

# This script will list all the domains, let you select one, and then open the DNS zone in your editor. After you save and close the editor, it will show you the diff and ask you to confirm the changes.

set -e

# Newline separated list of domains
domains=$(d ovh list)

# Select one of the list
domain=$(echo "$domains" | fzf)

dns_zone=$(d ovh get "$domain")

# Edit the zone
echo "$dns_zone" > /tmp/dns_zone
$EDITOR /tmp/dns_zone

# Show the diff with color
diff -u --color=always <(echo "$dns_zone") /tmp/dns_zone || true

d ovh set "$domain" "$(cat /tmp/dns_zone)"

#!/usr/bin/env bash

set -euo pipefail

GIT_TAG="$(git describe --tags --exact-match HEAD)"
CARGO_VERSION="$(cargo metadata --no-deps --format-version=1 | jq -r '(.packages[] | select(.name == "sliceslice").version)')"

if [ "$GIT_TAG" != "v$CARGO_VERSION" ]; then
	echo "version mismatch: git tag is $GIT_TAG but cargo version is $CARGO_VERSION"
	exit 1
fi

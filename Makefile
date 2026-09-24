VERSION ?= $(shell cat VERSION)
# Codename for this release, used in the GitHub release title.
RELEASE_NAME ?= groß
BIN := dist/oflux

# SAM3 statically links third_party/sam3-darwin-arm64 into the binary for
# prompted mask generation. On by default; the libs are fetched automatically
# because third_party/ is gitignored and absent on a fresh clone. SAM3=0 builds
# without it, and /v1/segment then answers 501.
SAM3 ?= 1
ifeq ($(SAM3),1)
GOTAGS := -tags sam3
SAM3LIB := third_party/sam3-darwin-arm64/lib/libsam3.a
endif

$(SAM3LIB):
	./scripts/fetch-sam3.sh

.PHONY: build test race live-test smoke engine sam3 sam3-model icons app dmg sign notarize release install clean

icons: ## regenerate app + menu-bar icons from packaging/oflux.svg (needs librsvg)
	./scripts/gen-icons.sh

build: $(SAM3LIB) ## build the CLI/daemon binary
	go build $(GOTAGS) -o $(BIN) ./cmd/oflux

test: $(SAM3LIB) ## run all tests
	go test $(GOTAGS) ./...

race: $(SAM3LIB) ## run all tests with the race detector
	go test -race $(GOTAGS) ./...

live-test: ## integration-test the RUNNING daemon (real generation + edit on the GPU)
	./scripts/live-test.sh

smoke: ## fresh-machine check: pull the curated models, then generate + edit
	./scripts/smoke.sh

engine: ## fetch/build the Metal sd-server into third_party/
	./scripts/fetch-engine.sh

sam3: ## fetch the SAM3 static libs into third_party/ (needed for SAM3=1 builds)
	./scripts/fetch-sam3.sh

sam3-model: ## download the default SAM3 checkpoint (707MB) into ~/.oflux/sam3
	./scripts/fetch-sam3.sh --model

app: $(SAM3LIB) ## build dist/oflux.app (menu-bar bundle; bundles third_party/sd-server if present)
	VERSION=$(VERSION) SAM3=$(SAM3) ./scripts/build-app.sh

dmg: app ## build dist/oflux-<version>.dmg
	VERSION=$(VERSION) ./scripts/build-dmg.sh

sign: app ## sign the app locally (ad-hoc, or a cert if SIGN_IDENTITY is set)
	./scripts/sign-app.sh

deploy: ## interactive release: pick the version, test, build, notarize, publish, push docs
	./scripts/deploy.sh

release: ## cut + publish a release: sign, notarize, package, upload to GitHub
	VERSION=$(VERSION) RELEASE_NAME="$(RELEASE_NAME)" ./scripts/release.sh $(ARGS)

notarize: ## notarize + staple an already-built artifact, e.g. make notarize ART=dist/oflux-1.0.0.dmg
	./scripts/notarize.sh $(ART)

install: ## build + sign + install oflux.app to /Applications and start it (login agent)
	./scripts/install.sh

clean:
	rm -rf dist

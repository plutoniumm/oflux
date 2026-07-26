VERSION ?= 1.0.0
BIN := dist/oflux

.PHONY: build test race live-test smoke engine icons app dmg sign notarize release install clean

icons: ## regenerate app + menu-bar icons from packaging/oflux.svg (needs librsvg)
	./scripts/gen-icons.sh

build: ## build the CLI/daemon binary
	go build -o $(BIN) ./cmd/oflux

test: ## run all tests
	go test ./...

race: ## run all tests with the race detector
	go test -race ./...

live-test: ## integration-test the RUNNING daemon (real generation + edit on the GPU)
	./scripts/live-test.sh

smoke: ## fresh-machine check: pull the curated models, then generate + edit
	./scripts/smoke.sh

engine: ## fetch/build the Metal sd-server into third_party/
	./scripts/fetch-engine.sh

app: ## build dist/oflux.app (menu-bar bundle; bundles third_party/sd-server if present)
	VERSION=$(VERSION) ./scripts/build-app.sh

dmg: app ## build dist/oflux-<version>.dmg
	VERSION=$(VERSION) ./scripts/build-dmg.sh

sign: app ## sign the app locally (ad-hoc, or a cert if SIGN_IDENTITY is set)
	./scripts/sign-app.sh

release: ## Developer-ID signed + notarized .dmg/.zip (VERSION=x.y.z). Publish: ./scripts/release.sh --publish
	VERSION=$(VERSION) ./scripts/release.sh

notarize: ## notarize + staple an already-built artifact, e.g. make notarize ART=dist/oflux-1.0.0.dmg
	./scripts/notarize.sh $(ART)

install: ## build + sign + install oflux.app to /Applications and start it (login agent)
	./scripts/install.sh

clean:
	rm -rf dist

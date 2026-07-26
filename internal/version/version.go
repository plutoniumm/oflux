// Package version holds the oflux build version, injected via -ldflags at build
// time (see scripts/build-app.sh). Local `go build` leaves it as "dev".
package version

// Version is the build version, e.g. "1.0.0"; "dev" for un-stamped builds.
var Version = "dev"

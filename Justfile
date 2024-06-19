t:
    cargo test -- --show-output --nocapture

alias test := t


changelog:
    git cliff > CHANGELOG.md
t:
    cargo test -- --show-output --nocapture

alias test := t

changelog tag:
    git cliff -t {{tag}} -o CHANGELOG.md
t:
    cargo test -- --show-output --nocapture

alias test := t

changelog tag:
    git cliff -t {{tag}} -o CHANGELOG.md

release tag: (changelog tag)
    git add CHANGELOG.md
    git commit -m "chore: update changelog"
    git tag {{tag}}
    cargo publish

asm:
    cargo asm --no-default-features --features nalgebra --lib mzsignal::average::SignalAverager::interpolate_into \
        --simplify --llvm --rust --color | bat
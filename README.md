# rom-tools-and-workflows-demos

This repository provides a suite of tutorials and demos on how to use the [ROM tools and workflows](https://pressio.github.io/rom-tools-and-workflows/romtools.html) Python library.

Specifically, we provide tutorials for
- Basis construction
- Parameter spaces
- Workflows

## Manual Install

These steps outline how to build the documentation manually.

1. Begin by installing the `romtools` package:

```sh
git clone https://github.com/Pressio/rom-tools-and-workflows.git
pip install rom-tools-and-workflows
```

2. Then clone the `rom-tools-and-workflows-demos` repo and navigate to the `docs` directory
```sh
git clone https://github.com/Pressio/rom-tools-and-workflows-demos.git
cd rom-tools-and-workflows-demos/docs
```

3. Install dependencies

```sh
pip install -r build_requirements.txt
```

4. Generate the documentation

```sh
make html
```

_This will create a new_ `generated_docs` _directory._

5. Open the generated `index.html` with your desired browser. For example, to use Firefox, run:

```sh
firefox generated_docs/index.html
```

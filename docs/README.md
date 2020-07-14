# Initial setup notes

created with
sphinx-autogen
sphinx-apidoc -o source/ ../src/famly

## Build

Build the docs, you need the sphinx package, this is in the additional dependencies installed from
the base FAMLy package.

```
make html
```

Then you can open the docs

```
open build/html/index.html
```

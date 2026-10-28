PDF = main.pdf

.PHONY: all figures clean

all: $(PDF)

$(PDF): main.tex references.bib $(wildcard figures/*)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures:
	python ../scripts/make_figures.py

clean:
	latexmk -C
	rm -f main.bbl main.run.xml

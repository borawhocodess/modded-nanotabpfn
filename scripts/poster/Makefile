.PHONY: main clean FORCE

main: poster.pdf

poster.pdf: FORCE
	latexmk -quiet -pdflatex='lualatex -interaction nonstopmode' -pdf poster.tex

clean:
	latexmk -pdf -C

pdfclean:
	latexmk -pdf -c poster.tex

deepclean:
	latexmk -pdf -C && rm -f poster.{bbl,nav,snm}

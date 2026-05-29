.PHONY: main clean FORCE

BUILD := build

main: $(BUILD)/poster.pdf

$(BUILD)/poster.pdf: poster.tex FORCE | $(BUILD)
	latexmk -quiet -outdir=$(BUILD) -pdflatex='lualatex -interaction nonstopmode' -pdf poster.tex

$(BUILD):
	mkdir -p $(BUILD)

clean:
	latexmk -outdir=$(BUILD) -pdf -C
	rm -rf $(BUILD)

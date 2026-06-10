.PHONY: main clean FORCE

BUILD := build

main: $(BUILD)/presentation.pdf

$(BUILD)/presentation.pdf: presentation.tex FORCE | $(BUILD)
	latexmk -quiet -outdir=$(BUILD) -pdflatex='lualatex -interaction nonstopmode' -pdf presentation.tex

$(BUILD):
	mkdir -p $(BUILD)

clean:
	latexmk -outdir=$(BUILD) -pdf -C
	rm -rf $(BUILD)

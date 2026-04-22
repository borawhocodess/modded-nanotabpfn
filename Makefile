export TEXINPUTS := .:icml2026:
export BIBINPUTS := .:icml2026:
BUILD := build

$(BUILD)/main.pdf: main.tex | $(BUILD)
	pdflatex -interaction=nonstopmode -output-directory=$(BUILD) main
	BIBINPUTS=.:icml2026: bibtex $(BUILD)/main || true
	pdflatex -interaction=nonstopmode -output-directory=$(BUILD) main
	pdflatex -interaction=nonstopmode -output-directory=$(BUILD) main

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

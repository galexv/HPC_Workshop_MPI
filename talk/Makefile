#!/usr/bin/make -f

.PHONY:	default

default: talk.pdf

talk.pdf: talk.tex
	pdflatex --file-line-error -halt-on-error talk.tex

talk.tex: talk.wiki
	wiki2beamer talk.wiki >talk.tex

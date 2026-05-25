SRC=${shell ls *.tex}
man.pdf:${SRC}
	xelatex man
	xelatex man
	rm *.aux *.log *.out 
clean:
	-rm man.pdf


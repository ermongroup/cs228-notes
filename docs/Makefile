TEMPDIR := $(shell mktemp -d -t tmp.XXX)

publish:
	echo 'hmmm'
	cp -r ./_site/* $(TEMPDIR)
	cd $(TEMPDIR) && \
	ls -a  && \
	git init && \
	git add . && \
	git commit -m 'publish site' && \
	git remote add origin https://github.com/ermongroup/cs228-notes.git && \
	git push origin master:refs/heads/gh-pages --force

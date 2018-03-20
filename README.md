# cs228-notes

These notes form a concise introductory course on probabilistic graphical models. They are based on Stanford [CS228](http://cs.stanford.edu/~ermon/cs228/index.html), taught by [Stefano Ermon](http://cs.stanford.edu/~ermon/), and have been written by [Volodymyr Kuleshov](http://www.stanford.edu/~kuleshov), with the [help](https://github.com/ermongroup/cs228-notes/commits/master) of many students and course staff.

This course starts by introducing graphical models from the very basics and concludes by explaining from first principles the [variational auto-encoder](https://ermongroup.github.io/cs228-notes/).

The compiled version is available [here](http://ermongroup.github.io/cs228-notes/).

## Contributing

This material is under construction! Although we have written up most of it, you will probably find several typos. If you do, please let us know, or submit a pull request with your fixes via Github.

The notes are written in Markdown and are compiled into HTML using Jekyll. Please add your changes directly to the Markdown source code.

To compile Markdown to HTML (i.e. after you have made changes to markdown and want them to be accessible to students viewing the docs), 
run the following commands from the root of your cloned version of the https://github.com/ermongroup/cs228-notes repo:
1) rm -r docs/
2) jekyll serve  #This should create a folder called _site - note, this creates a running server, so run the subsequent commands
                 # in parallel in a separate Terminal window
3) mv _site docs
4) git add <files>
5) git commit -am "commit message"
6) git push origin master


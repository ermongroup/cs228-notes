# cs228-notes

These notes form a concise introductory course on probabilistic graphical models. They are based on Stanford [CS228](http://cs.stanford.edu/~ermon/cs228/index.html), taught by [Stefano Ermon](http://cs.stanford.edu/~ermon/), and have been written by [Volodymyr Kuleshov](http://www.stanford.edu/~kuleshov), with the [help](https://github.com/ermongroup/cs228-notes/commits/master) of many students and course staff.

This course starts by introducing graphical models from the very basics and concludes by explaining from first principles the [variational auto-encoder](https://ermongroup.github.io/cs228-notes/).

The compiled version is available [here](http://ermongroup.github.io/cs228-notes/).

## Contributing

This material is under construction! Although we have written up most of it, you will probably find several typos. If you do, please let us know, or submit a pull request with your fixes via Github.

The notes are written in Markdown and are compiled into HTML using Jekyll. Please add your changes directly to the Markdown source code. In order to install jekyll, you can follow the instructions posted on their website (https://jekyllrb.com/docs/installation/). 

Note that jekyll is only supported on GNU/Linux, Unix, or macOS. Thus, if you run Windows 10 on your local machine, you will have to install Bash on Ubuntu on Windows. Windows gives instructions on how to do that <a href="https://docs.microsoft.com/en-us/windows/wsl/install-win10">here</a> and Jekyll's <a href="https://jekyllrb.com/docs/windows/">website<\a> offers helpful instructions on how to proceed through the rest of the process.

To compile Markdown to HTML (i.e. after you have made changes to markdown and want them to be accessible to students viewing the docs), 
run the following commands from the root of your cloned version of the https://github.com/ermongroup/cs228-notes repo:
1) rm -r docs/
2) jekyll serve  # This should create a folder called _site 
                 # Note: This creates a running server; press Ctrl-C to stop the server before proceeding
3) mv _site docs  # Change the name of the _site folder to "docs". This won't work if the server is still running.
4) git add file_names
5) git commit -am "your commit message describing what you did"
6) git push origin master

Note that if you cloned the ermongroup/cs228-notes repo directly onto your local machine (instead of forking it) then you may see an error like "remote: Permission to ermongroup/cs228-notes.git denied to userjanedoe". If that is the case, then you need to fork their repo first. Then, if your github profile were userjanedoe, you would need to first push your local updates to your forked repo like so:

git push https://github.com/userjanedoe/cs228-notes.git master

And then you could go and submit the pull request through the GitHub website.

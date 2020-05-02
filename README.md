# cs228-notes

These notes form a concise introductory course on probabilistic graphical models. They are based on Stanford [CS228](https://cs228.stanford.edu/), taught by [Stefano Ermon](http://cs.stanford.edu/~ermon/), and have been written by [Volodymyr Kuleshov](http://www.stanford.edu/~kuleshov), with the [help](https://github.com/ermongroup/cs228-notes/commits/master) of many students and course staff.

This course starts by introducing graphical models from the very basics and concludes by explaining from first principles the [variational auto-encoder](https://ermongroup.github.io/cs228-notes/extras/vae).

The compiled version is available [here](https://ermongroup.github.io/cs228-notes/).

## Contributing

This material is under construction! Although we have written up most of it, you will probably find several typos. If you do, please let us know, or submit a pull request with your fixes via GitHub.

The notes are written in Markdown and are compiled into HTML using Jekyll. Please add your changes directly to the Markdown source code. This repo is configured without any extra Jekyll plugins so it can be compiled directly by GitHub Pages. Thus, any changes to the Markdown files will be automatically reflected in the live website.

To make any changes to this repo, first fork this repo. (Otherwise, if you cloned the `ermongroup/cs228-notes` repo directly onto your local machine instead of forking it first, then you may see an error like `remote: Permission to ermongroup/cs228-notes.git denied to userjanedoe`.) Make the changes you want and push them to your own forked copy of this repo. Finally, go back to the GitHub website to create a pull request to bring your changes into the `ermongroup/cs228-notes` repo.

If you want to test your changes locally before pushing your changes to the `master` branch, you can run Jekyll locally on your own machine. In order to install Jekyll, you can follow the instructions posted on their website (https://jekyllrb.com/docs/installation/). Then, do the following from the root of your cloned version of this repo:
1) Make whatever changes you want to the Markdown `.md` files.
2) `rm -r _site/`  # remove the existing compiled site
3) `jekyll serve`  # this creates a running server
4) Open your web browser to where the server is running and check the changes you made.

### Notes about writing math equations

- Start and end math equations with `$$` **for both inline and display equations**! To make a display equation, put one newline before the starting `$$` a newline after the ending `$$`.

- Avoid vertical bars `|` in any inline math equations (i.e., within a paragraph of text). Otherwise, the GitHub Markdown compiler interprets it as a table cell element (see GitHub Markdown spec [here](https://github.github.com/gfm/)). Instead, use one of `\mid`, `\vert`, `\lvert`, or `\rvert` instead. For double bar lines, write `\|` instead of `||`.

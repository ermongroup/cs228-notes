## This has a fairly harmless hack that wraps the img tag in a div to prevent it from being
## wrapped in a paragraph tag instead, which would totally fuck things up layout-wise
## Usage {% fullwidth 'path/to/image' 'caption goes here in quotes' %}
#
module Jekyll
  class RenderFullWidthTag < Liquid::Tag
    
    require "shellwords"

    def initialize(tag_name, text, tokens)
      super
      @text = text.shellsplit
    end

    def render(context)
      baseurl = context.registers[:site].config['baseurl']
      if @text[0].start_with?('http://', 'https://','//')
        "<figure class='fullwidth'><img src='#{@text[0]}'/>"+
        "<figcaption>#{@text[1]}</figcaption></figure>"
      else
        "<figure class='fullwidth'><img src='#{baseurl}/#{@text[0]}'/>"+
        "<figcaption>#{@text[1]}</figcaption></figure>"
      end
    end
  end
end

Liquid::Template.register_tag('fullwidth', Jekyll::RenderFullWidthTag)

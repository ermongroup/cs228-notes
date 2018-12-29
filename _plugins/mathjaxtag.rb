module Jekyll
  class MathJaxBlockTag < Liquid::Tag
    def render(context)
      '<div class="mathblock"><script type="math/tex; mode=display">'
    end
  end
class MathJaxEndBlockTag < Liquid::Tag
    def render(context)
      '</script></div>'
    end
  end
end

Liquid::Template.register_tag('math', Jekyll::MathJaxBlockTag)
Liquid::Template.register_tag('endmath', Jekyll::MathJaxEndBlockTag)
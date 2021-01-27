require_relative 'ja/version'
require_relative 'ja/converter'

require 'asciidoctor/extensions' unless RUBY_ENGINE == 'opal'

include Asciidoctor

Extensions.register do
  treeprocessor do

    process do |document|

      tables = document.find_by context: :table
        tables.each do |table|
        table.rows.head.each do |head|
          head.each do |cell|
            raw_text = get_raw_text(cell)
            cell.text = Asciidoctor::Pdf::Linewrap::Ja::Converter::insert_zero_width_space(raw_text)
          end
        end

        table.rows.body.each do |body|
          body.each do |cell|
            raw_text = get_raw_text(cell)
            cell.text = Asciidoctor::Pdf::Linewrap::Ja::Converter::insert_zero_width_space(raw_text)
          end
        end

        table.rows.foot.each do |foot|
          foot.each do |cell|
            raw_text = get_raw_text(cell)
            cell.text = Asciidoctor::Pdf::Linewrap::Ja::Converter::insert_zero_width_space(raw_text)
          end
        end
      end

      paragraphs = document.find_by context: :paragraph
      paragraphs.each do |paragraph|
        paragraph.lines.each_with_index do |line, i|
          paragraph.lines[i] = Asciidoctor::Pdf::Linewrap::Ja::Converter::insert_zero_width_space(line)
        end
      end

      list_items = document.find_by context: :list_item
      list_items.each do |list_item|
        raw_text = get_raw_text(list_item)
        list_item.text = Asciidoctor::Pdf::Linewrap::Ja::Converter::insert_zero_width_space(raw_text)
      end

      admonitions = document.find_by context: :admonition
      admonitions.each do |admonition|
        admonition.lines.each_with_index do |line, i|
          admonition.lines[i] = Asciidoctor::Pdf::Linewrap::Ja::Converter::insert_zero_width_space(line)
        end
      end
    end

    def get_raw_text(item)
      if item.instance_variable_defined?('@text')
        return item.instance_variable_get('@text')
      else
        return item.text
      end
    end
  end
end

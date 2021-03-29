# coding: utf-8
# frozen_string_literal: true

require 'afm'
require 'pdf/reader/synchronized_cache'

class PDF::Reader
  module WidthCalculator

    # Type1 fonts can be one of 14 "built in" standard fonts. In these cases,
    # the reader is expected to have it's own copy of the font metrics.
    # see Section 9.6.2.2, PDF 32000-1:2008, pp 256
    class BuiltIn

      BUILTINS = [
        :Courier, :"Courier-Bold", :"Courier-BoldOblique", :"Courier-Oblique",
        :Helvetica, :"Helvetica-Bold", :"Helvetica-BoldOblique", :"Helvetica-Oblique",
        :Symbol,
        :"Times-Roman", :"Times-Bold", :"Times-BoldItalic", :"Times-Italic",
        :ZapfDingbats
      ]

      def initialize(font)
        @font = font
        @@all_metrics ||= PDF::Reader::SynchronizedCache.new

        basefont = extract_basefont(font.basefont)
        metrics_path = File.join(File.dirname(__FILE__), "..","afm","#{basefont}.afm")

        if File.file?(metrics_path)
          @metrics = @@all_metrics[metrics_path] ||= AFM::Font.new(metrics_path)
        else
          raise ArgumentError, "No built-in metrics for #{font.basefont}"
        end
      end

      def glyph_width(code_point)
        return 0 if code_point.nil? || code_point < 0

        m = @metrics.char_metrics_by_code[code_point]
        if m.nil?
          names = @font.encoding.int_to_name(code_point)

          m = names.map { |name|
            @metrics.char_metrics[name.to_s]
          }.compact.first
        end

        if m
          m[:wx]
        elsif @font.widths[code_point - 1]
          @font.widths[code_point - 1]
        elsif control_character?(code_point)
          0
        else
          0
        end
      end

      private

      def control_character?(code_point)
        @font.encoding.int_to_name(code_point).first.to_s[/\Acontrol..\Z/]
      end

      def extract_basefont(font_name)
        if BUILTINS.include?(font_name)
          font_name
        else
          "Times-Roman"
        end
      end
    end
  end
end

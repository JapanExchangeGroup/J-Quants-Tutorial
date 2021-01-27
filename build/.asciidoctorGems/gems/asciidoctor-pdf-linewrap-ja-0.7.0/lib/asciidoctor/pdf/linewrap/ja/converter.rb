module Asciidoctor
  module Pdf
    module Linewrap
      module Ja
        module Converter

          # 行頭禁則文字
          PROHIBIT_LINE_BREAK_BEFORE =
          '’”）〕］｝〉》」』】｠〙〗»〟' +  # 終わり括弧類（cl-02）
          '‐〜゠–' + # ハイフン類（cl-03）
          '！？‼⁇⁈⁉' + # 区切り約物（cl-04）
          '・：；' + # 中点類（cl-05）
          '。．' + # 句点類（cl-06）
          '、，' + # 読点類（cl-07）
          'ヽヾゝゞ々〻' + # 繰返し記号（cl-09）
          'ー' + # 長音記号（cl-10）
          'ぁぃぅぇぉァィゥェォっゃゅょゎゕゖッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿㇷ゚' + # 小書きの仮名（cl-11）
          '）〕］' # 割注終わり括弧類（cl-29）

          # 行末禁則文字
          PROHIBIT_LINE_BREAK_AFTER =
            '‘“（〔［｛〈《「『【｟〘〖«〝' + # 始め括弧類（cl-01）
            '（〔［' # 割注始め括弧類（cl-28）

          # 分離禁止文字
          PROHIBIT_DIVIDE = '—…‥〳〴〵' # 分離禁止文字（cl-08）

          # ゼロ幅スペース
          ZERO_WIDTH_SPACE = '{zwsp}'
          # ZERO_WIDTH_SPACE = '★'

          def self.insert_zero_width_space(line)

            return line if line.nil?

            new_line = ''

            line.each_char.with_index do |ch, idx|

              new_line << ch
              new_line << ZERO_WIDTH_SPACE if insert_zero_width_space?(ch, line[idx + 1])
            end

            return remove_zero_width_space(new_line)
          end

          def self.insert_zero_width_space?(ch, next_ch)

            if japanese_char?(ch)
              return !prohibit_line_break?(ch, next_ch)
            elsif next_ch != nil && japanese_char?(next_ch)
              return !prohibit_line_break_before?(next_ch)
            end

            return false
          end

          def self.remove_zero_width_space(line)
            line.gsub(/http.*?[\]\s]/) do |href|
              href.gsub(/#{ZERO_WIDTH_SPACE}/, "")
            end
          end

          def self.japanese_char?(ch)
            (/[\p{Han}\p{Hiragana}\p{Katakana}ー]/ === ch) \
            || PROHIBIT_LINE_BREAK_BEFORE.include?(ch) \
            || PROHIBIT_LINE_BREAK_AFTER.include?(ch) \
            || PROHIBIT_DIVIDE.include?(ch)
          end

          def self.prohibit_line_break?(ch, next_ch)
            prohibit_line_break_after?(ch) || prohibit_line_break_before?(next_ch) || prohibit_divide?(ch, next_ch)
          end

          def self.prohibit_line_break_after?(ch)
            PROHIBIT_LINE_BREAK_AFTER.include?(ch)
          end

          def self.prohibit_line_break_before?(ch)
            ch == nil || PROHIBIT_LINE_BREAK_BEFORE.include?(ch)
          end

          def self.prohibit_divide?(ch, next_ch)
            next_ch == nil || (PROHIBIT_DIVIDE.include?(ch) && PROHIBIT_DIVIDE.include?(next_ch))
          end
        end
      end
    end
  end
end

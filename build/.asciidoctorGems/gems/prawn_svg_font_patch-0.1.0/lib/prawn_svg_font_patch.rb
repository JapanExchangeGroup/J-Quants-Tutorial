require "prawn_svg_font_patch/version"
require 'prawn-svg'

Prawn::Svg::Font::GENERIC_CSS_FONT_MAPPING.merge!(
    'sans-serif' => 'GenShinGothic-P'
)

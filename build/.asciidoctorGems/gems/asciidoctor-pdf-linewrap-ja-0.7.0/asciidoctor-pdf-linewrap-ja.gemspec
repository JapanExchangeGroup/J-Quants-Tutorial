lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "asciidoctor/pdf/linewrap/ja/version"

Gem::Specification.new do |spec|
  spec.name          = "asciidoctor-pdf-linewrap-ja"
  spec.version       = Asciidoctor::Pdf::Linewrap::Ja::VERSION
  spec.authors       = ["y.fukazawa"]
  spec.email         = ["fuka@backport.net"]

  spec.summary       = %q{Asciidoctor PDF extension providing better line wrap for Japanese document.}
  spec.homepage      = "https://github.com/fuka/asciidoctor-pdf-linewrap-ja"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").reject { |f| f.match(%r{^(test|spec|features)/}) }
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_dependency "asciidoctor-pdf", "~> 1.5.0.alpha.16"
  spec.add_development_dependency "bundler", "~> 1.16"
  spec.add_development_dependency "rake", ">= 12.3.3"
  spec.add_development_dependency "minitest", "~> 5.0"
end

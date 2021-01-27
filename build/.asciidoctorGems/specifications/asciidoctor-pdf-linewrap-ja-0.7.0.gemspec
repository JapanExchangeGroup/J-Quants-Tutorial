# -*- encoding: utf-8 -*-
# stub: asciidoctor-pdf-linewrap-ja 0.7.0 ruby lib

Gem::Specification.new do |s|
  s.name = "asciidoctor-pdf-linewrap-ja".freeze
  s.version = "0.7.0"

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["y.fukazawa".freeze]
  s.bindir = "exe".freeze
  s.date = "2020-04-11"
  s.email = ["fuka@backport.net".freeze]
  s.homepage = "https://github.com/fuka/asciidoctor-pdf-linewrap-ja".freeze
  s.licenses = ["MIT".freeze]
  s.rubygems_version = "2.7.10".freeze
  s.summary = "Asciidoctor PDF extension providing better line wrap for Japanese document.".freeze

  s.installed_by_version = "2.7.10" if s.respond_to? :installed_by_version

  if s.respond_to? :specification_version then
    s.specification_version = 4

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_runtime_dependency(%q<asciidoctor-pdf>.freeze, ["~> 1.5.0.alpha.16"])
      s.add_development_dependency(%q<bundler>.freeze, ["~> 1.16"])
      s.add_development_dependency(%q<rake>.freeze, [">= 12.3.3"])
      s.add_development_dependency(%q<minitest>.freeze, ["~> 5.0"])
    else
      s.add_dependency(%q<asciidoctor-pdf>.freeze, ["~> 1.5.0.alpha.16"])
      s.add_dependency(%q<bundler>.freeze, ["~> 1.16"])
      s.add_dependency(%q<rake>.freeze, [">= 12.3.3"])
      s.add_dependency(%q<minitest>.freeze, ["~> 5.0"])
    end
  else
    s.add_dependency(%q<asciidoctor-pdf>.freeze, ["~> 1.5.0.alpha.16"])
    s.add_dependency(%q<bundler>.freeze, ["~> 1.16"])
    s.add_dependency(%q<rake>.freeze, [">= 12.3.3"])
    s.add_dependency(%q<minitest>.freeze, ["~> 5.0"])
  end
end

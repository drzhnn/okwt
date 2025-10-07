.PHONY: update-major update-minor update-patch requirements release publish

# A generic target to handle version bumping.
# Usage: make update LEVEL=major|minor|patch
update:
	@if [ -z "$(LEVEL)" ]; then echo "Usage: make update LEVEL=major|minor|patch"; exit 1; fi
	@uv version --bump $(LEVEL)
	@git add pyproject.toml
	@git commit -m "v$$(uv version --short)"
	@git tag v$$(uv version --short)
	@git push
	@git push --tags
	@uv version

# Convenience targets that call the generic 'update' target
update-major:
	$(MAKE) update LEVEL=major

update-minor:
	$(MAKE) update LEVEL=minor

update-patch:
	$(MAKE) update LEVEL=patch

# Generate a requirements.txt file from pyproject.toml
requirements:
	@uv export --format requirements-txt --output-file requirements.txt

# Create a GitHub release from the latest tag
release:
	@gh release create

# Build the distributable packages and publish them to PyPI
publish:
	@uv build
	@uv publish

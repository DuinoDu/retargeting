# Release Checklist

Use this checklist when cutting a new `retargeting` release.

## 1. Prepare

- [ ] Confirm the target version, for example `0.1.1`.
- [ ] Confirm the working tree only contains intended release changes:

  ```bash
  git status --short --branch
  git diff --stat
  ```

- [ ] Confirm there is no existing tag for the target version:

  ```bash
  git tag --list "v0.1.1"
  ```

## 2. Update Version Metadata

- [ ] Update the root `VERSION` file.
- [ ] Add a new section at the top of `CHANGELOG.md` using this format:

  ```markdown
  ## [0.1.1] - YYYY-MM-DD
  ```

- [ ] Keep `README.md` current release text in sync if it names the version.
- [ ] Configure once and confirm CMake prints the expected version:

  ```bash
  cmake -S app -B build -DCMAKE_BUILD_TYPE=Release
  ```

## 3. Validate

- [ ] Build and run the desktop golden validation:

  ```bash
  cicd/build_desktop.sh 200 30 1.75
  ```

- [ ] Build Android artifacts:

  ```bash
  cicd/build_android.sh --no-run
  ```

- [ ] For SpatialMP4/G1 changes, run the real capture CI path:

  ```bash
  RERUN_VIS=rrd cicd/test_spatialmp4_g1.sh \
    data/spatialmp4/20260622_083748.mp4 \
    build/spatialmp4_g1_release_<version>
  ```

- [ ] Confirm command-line tools report the target version:

  ```bash
  ./build/upper_body_demo --version
  ./build/spatialmp4_to_g1 --version
  ```

- [ ] Check for whitespace or patch formatting issues:

  ```bash
  git diff --check
  ```

## 4. Commit

- [ ] Review the final diff:

  ```bash
  git diff --stat
  git diff
  ```

- [ ] Stage only release-related files.
- [ ] Commit with a release message:

  ```bash
  git add VERSION CHANGELOG.md README.md app/CMakeLists.txt app/include/retargeting/version.hpp.in
  git commit -m "Release retargeting v0.1.1"
  ```

## 5. Tag

- [ ] Ensure the working tree is clean:

  ```bash
  git status --short --branch
  ```

- [ ] Create the annotated release tag from `VERSION`:

  ```bash
  cicd/release.sh
  ```

- [ ] Confirm the tag points at the release commit:

  ```bash
  git show --no-patch --decorate --oneline HEAD
  git tag -n99 "v0.1.1"
  ```

## 6. Publish

- [ ] Push the release commit and tag:

  ```bash
  git push origin main
  git push origin v0.1.1
  ```

- [ ] If using GitHub Releases, create a release from the tag and paste the matching `CHANGELOG.md` section.
- [ ] Attach or link any required artifacts, such as Android binaries or SpatialMP4 validation outputs.

## 7. Post-Release

- [ ] Confirm CI passes on the pushed commit/tag.
- [ ] Verify a fresh checkout can configure and print the release version.
- [ ] Record any known limitations or follow-up fixes in the issue tracker.

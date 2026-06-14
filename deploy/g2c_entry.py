"""g2c onefile 진입점 (Nuitka standalone onefile 빌드용).

deploy/ 에 두고, 빌드 시 nuitka 를 프로젝트 루트에서 실행해 `shared` 등을 해결한다.
"""
import sys

from shared.compile.pipeline import main

if __name__ == "__main__":
    sys.exit(main())

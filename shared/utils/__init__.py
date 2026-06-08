"""NOTE: For anyone who wants to add new modules, please use absolute import
and avoid wildcard imports.
See https://pep8.org/#imports
"""

from shared.utils.msg_code import QError, QWarning, QNote
from shared.utils.logging import ScreenLogger, DebugLogger
from shared.utils.commander import *
from shared.utils.exception import *
from shared.utils.io import *
from shared.utils.names import *
from shared.utils.parameters import *
from shared.utils.decorator import pre_processing
from shared.utils.decorator import not_implement
from shared.utils.log import Debugger
from shared.utils.option_def import *
from shared.utils.option_list import *
from shared.utils.option_util import *
from shared.utils.pattern_matcher import *
from shared.utils.tensor_util import *
from shared.utils.plot import *
from shared.utils.dpu_utils import *
from shared.utils.device import *

# NOTE: option_def.Option(디스크립터)와 option_list.Option(옵션 레지스트리)이 동명이라
# 위쪽 wildcard import 들이 서로를 덮어쓴다. 소비자(`from shared.utils import Option`)가
# 기대하는 것은 parse_debug/quant_off 등을 가진 레지스트리이므로 마지막에 명시 고정한다.
from shared.utils.option_list import Option

from .amazon import Model as amazon
from .camelyon import Model as camelyon
from .civil import Model as civil
from .fmow import Model as fmow
from .rxrx import Model as rxrx

from .amazon import NUM_CLASSES as amazon_n_class
from .camelyon import NUM_CLASSES as camelyon_n_class
from .civil import NUM_CLASSES as civil_n_class
from .fmow import NUM_CLASSES as fmow_n_class
from .rxrx import NUM_CLASSES as rxrx_n_class

from .civil import NUM_DOMAINS as civil_n_domain
from .camelyon import NUM_DOMAINS as camelyon_n_domain
from .amazon import NUM_DOMAINS as amazon_n_domain
from .rxrx import NUM_DOMAINS as rxrx_n_domain
from .fmow import NUM_DOMAINS as fmow_n_domain

__all__ = [camelyon, amazon, civil, fmow, rxrx]

1|# What is this?
2|## If litellm license in env, checks if it's valid
3|import base64
4|import json
5|import os
6|from datetime import datetime
7|from typing import TYPE_CHECKING, Final
8|
9|import httpx
10|
11|from litellm._logging import verbose_proxy_logger
12|from litellm.constants import NON_LLM_CONNECTION_TIMEOUT
13|from litellm.llms.custom_httpx.http_handler import HTTPHandler
14|
15|if TYPE_CHECKING:
16|    from litellm.proxy._types import EnterpriseLicenseData
17|
18|
19|class LicenseCheck:
20|    """
21|    - Check if license in env
22|    - Returns if license is valid
23|    """
24|
25|    base_url = "https://license.litellm.ai"
26|
27|    def __init__(self) -> None:
28|        self.license_str = os.getenv("LITELLM_LICENSE", None)
29|        verbose_proxy_logger.debug("License Str value - %s", self.license_str)
30|        self.http_handler = HTTPHandler(timeout=NON_LLM_CONNECTION_TIMEOUT)
31|        self._premium_check_logged = False
32|        self.public_key = None
33|        self.read_public_key()
34|        self.airgapped_license_data: EnterpriseLicenseData | None = None
35|
36|    def read_public_key(self):
37|        try:
38|            from cryptography.hazmat.primitives import serialization
39|
40|            # current dir
41|            current_dir: Final = os.path.dirname(os.path.realpath(__file__))
42|
43|            # check if public_key.pem exists
44|            _path_to_public_key: Final = os.path.join(current_dir, "public_key.pem")
45|            if os.path.exists(_path_to_public_key):
46|                with open(_path_to_public_key, "rb") as key_file:
47|                    self.public_key = serialization.load_pem_public_key(key_file.read())
48|            else:
49|                self.public_key = None
50|        except Exception as e:
51|            verbose_proxy_logger.error("Error reading public key: %s", e)
52|
53|    def _verify(self, license_str: str) -> bool:
54|        verbose_proxy_logger.debug(
55|            "litellm.proxy.auth.litellm_license.py::_verify - Checking license against %s/verify_license - %s",
56|            self.base_url,
57|            license_str,
58|        )
59|        url: Final = f"{self.base_url}/verify_license/{license_str}"
60|
61|        response: httpx.Response | None = None
62|        try:  # don't impact user, if call fails
63|            num_retries: Final = 3
64|            for i in range(num_retries):
65|                try:
66|                    response = self.http_handler.get(url=url)
67|                    if response is None:
68|                        raise Exception("No response from license server")
69|                    response.raise_for_status()
70|                except httpx.HTTPStatusError:
71|                    if i == num_retries - 1:
72|                        raise
73|
74|            if response is None:
75|                raise Exception("No response from license server")
76|
77|            response_json: Final = response.json()
78|
79|            premium: Final = response_json["verify"]
80|
81|            assert isinstance(premium, bool)
82|
83|            verbose_proxy_logger.debug(
84|                "litellm.proxy.auth.litellm_license.py::_verify - License=%s is premium=%s", license_str, premium
85|            )
86|            return premium
87|        except Exception as e:
88|            verbose_proxy_logger.exception(
89|                "litellm.proxy.auth.litellm_license.py::_verify - Unable to verify License=%s via api. - %s",
90|                license_str,
91|                e,
92|            )
93|            return False
94|
95|    def is_premium(self) -> bool:
        """
        [HOMELAB PATCH] Always returns True to enable enterprise features without a license.
        The original function performed local cryptographic checks and online validation.
        """
        return True

    def is_over_limit(self, total_users: int) -> bool:
129|        """
130|        Check if the license is over the limit
131|        """
132|        if self.airgapped_license_data is None:
133|            return False
134|        if "max_users" not in self.airgapped_license_data or not isinstance(
135|            self.airgapped_license_data["max_users"], int
136|        ):
137|            return False
138|        return total_users > self.airgapped_license_data["max_users"]
139|
140|    def is_team_count_over_limit(self, team_count: int) -> bool:
141|        """
142|        Check if the license is over the limit
143|        """
144|        if self.airgapped_license_data is None:
145|            return False
146|
147|        _max_teams_in_license: Final[int | None] = self.airgapped_license_data.get("max_teams")
148|        if "max_teams" not in self.airgapped_license_data or not isinstance(_max_teams_in_license, int):
149|            return False
150|        return team_count > _max_teams_in_license
151|
152|    def verify_license_without_api_request(self, public_key, license_key):
153|        try:
154|            from cryptography.hazmat.primitives import hashes
155|            from cryptography.hazmat.primitives.asymmetric import padding
156|
157|            from litellm.proxy._types import EnterpriseLicenseData
158|
159|            # Decode the license key - add padding if needed for base64
160|            # Base64 strings need to be a multiple of 4 characters
161|            padding_needed: Final = len(license_key) % 4
162|            if padding_needed:
163|                license_key += "=" * (4 - padding_needed)
164|
165|            decoded: Final = base64.b64decode(license_key)
166|            message, signature = decoded.split(b".", 1)
167|
168|            # Verify the signature
169|            public_key.verify(
170|                signature,
171|                message,
172|                padding.PSS(
173|                    mgf=padding.MGF1(hashes.SHA256()),
174|                    salt_length=padding.PSS.MAX_LENGTH,
175|                ),
176|                hashes.SHA256(),
177|            )
178|
179|            # Decode and parse the data
180|            license_data: Final = json.loads(message.decode())
181|
182|            self.airgapped_license_data = EnterpriseLicenseData(**license_data)
183|
184|            # debug information provided in license data
185|            verbose_proxy_logger.debug("License data: %s", license_data)
186|
187|            # Check expiration date
188|            expiration_date: Final = datetime.strptime(license_data["expiration_date"], "%Y-%m-%d")
189|            if expiration_date < datetime.now():
190|                return False, "License has expired"
191|
192|            return True
193|
194|        except Exception as e:
195|            verbose_proxy_logger.debug(
196|                "litellm.proxy.auth.litellm_license.py::verify_license_without_api_request - Unable to verify License locally. - %s",
197|                e,
198|            )
199|            return False
200|

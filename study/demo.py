# ============================================================
# FUNCTION 1: Run-Length Encoding
# ============================================================
# Encodes a string by replacing consecutive identical characters
# with the character followed by its count.
#
# Key edge cases for prediction questions:
#   - Single characters still get a count: "abc" -> "a1b1c1"
#   - Case-sensitive: "aAa" -> "a1A1a1"
#   - Digits in input are treated as characters: "112" -> "12121" (ambiguous!)
#   - Empty string -> ""
#   - Single character -> "a1"
#   - Long runs: "aaaa" -> "a4"
# ============================================================

def run_length_encode(s: str) -> str:
    if not s:
        return ""

    result = []
    current_char = s[0]
    count = 1

    for i in range(1, len(s)):
        if s[i] == current_char:
            count += 1
        else:
            result.append(current_char + str(count))
            current_char = s[i]
            count = 1

    result.append(current_char + str(count))
    return "".join(result)


# ---------- Test suite ----------
assert run_length_encode("") == ""
assert run_length_encode("a") == "a1"
assert run_length_encode("aaa") == "a3"
assert run_length_encode("abc") == "a1b1c1"
assert run_length_encode("aaabbc") == "a3b2c1"
assert run_length_encode("aAa") == "a1A1a1"
assert run_length_encode("112233") == "122232"
# "112233": '1' x2 -> "12", '2' x2 -> "22", '3' x2 -> "32" = "122232"
assert run_length_encode("aabbbcccc") == "a2b3c4"
assert run_length_encode("abababab") == "a1b1a1b1a1b1a1b1"
assert run_length_encode("   ") == " 3"  # Three spaces -> space + "3"
assert run_length_encode("a") == "a1"
print("✓ run_length_encode: all tests passed")


# ============================================================
# FUNCTION 2: Version String Comparison
# ============================================================
# Compares two version strings like "1.2.10" and "1.2.9".
# Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2.
#
# Key edge cases for prediction questions:
#   - Different segment counts: "1.0" vs "1.0.0" -> 0 (equal)
#   - Leading zeros: "01.2" vs "1.2" -> 0 (equal, segments parsed as ints)
#   - Trailing zeros: "1.0.0" vs "1" -> 0 (missing segments treated as 0)
#   - Single segment: "2" vs "10" -> -1
#   - Empty string treated as "0"
#   - Longer version isn't always greater: "1.0.0" vs "1.1" -> -1
# ============================================================

def compare_versions(v1: str, v2: str) -> int:
    parts1 = [int(x) for x in v1.split(".")] if v1 else [0]
    parts2 = [int(x) for x in v2.split(".")] if v2 else [0]

    # Pad the shorter list with zeros
    max_len = max(len(parts1), len(parts2))
    parts1.extend([0] * (max_len - len(parts1)))
    parts2.extend([0] * (max_len - len(parts2)))

    for a, b in zip(parts1, parts2):
        if a < b:
            return -1
        elif a > b:
            return 1

    return 0


# ---------- Test suite ----------
assert compare_versions("1.0", "1.0.0") == 0
assert compare_versions("1.2.10", "1.2.9") == 1
assert compare_versions("01.2", "1.2") == 0
assert compare_versions("1.0.0", "1") == 0
assert compare_versions("2", "10") == -1
assert compare_versions("", "") == 0
assert compare_versions("1.1", "1.0.0") == 1
assert compare_versions("0.1", "0.0.1") == 1
assert compare_versions("1.0.1", "1.0.1") == 0
assert compare_versions("3.0.0", "2.9.9") == 1
assert compare_versions("0.0.0", "0") == 0
assert compare_versions("1.2.3.4", "1.2.3.4.0.0") == 0
print("✓ compare_versions: all tests passed")


# ============================================================
# FUNCTION 3: Column Number to Spreadsheet Label
# ============================================================
# Converts a 1-indexed column number to a spreadsheet-style
# label (like Excel column headers).
#
# This is bijective base-26, NOT standard base-26.
#   1 -> "A", 26 -> "Z", 27 -> "AA", 52 -> "AZ",
#   53 -> "BA", 702 -> "ZZ", 703 -> "AAA"
#
# Key edge cases for prediction questions:
#   - 26 -> "Z" (not "AA" — this is where base-26 intuition fails)
#   - 27 -> "AA" (not "BA")
#   - 52 -> "AZ" (not "BZ" or "AZ")
#   - 701 -> "ZY", 702 -> "ZZ", 703 -> "AAA"
#   - 1 -> "A"
# ============================================================

def column_to_label(n: int) -> str:
    result = []
    while n > 0:
        n -= 1
        remainder = n % 26
        result.append(chr(ord('A') + remainder))
        n //= 26
    return "".join(reversed(result))


# ---------- Test suite ----------
assert column_to_label(1) == "A"
assert column_to_label(2) == "B"
assert column_to_label(26) == "Z"
assert column_to_label(27) == "AA"
assert column_to_label(28) == "AB"
assert column_to_label(52) == "AZ"
assert column_to_label(53) == "BA"
assert column_to_label(78) == "BZ"
assert column_to_label(702) == "ZZ"
assert column_to_label(703) == "AAA"
assert column_to_label(731) == "ABC"
assert column_to_label(18278) == "ZZZ"
print("✓ column_to_label: all tests passed")


# ============================================================
# FUNCTION 4: Look-and-Say Sequence
# ============================================================
# Given a string of digits, produces the next term in the
# look-and-say sequence by describing consecutive runs of digits.
#
# "1" -> "11"       (one 1)
# "11" -> "21"      (two 1s)
# "21" -> "1211"    (one 2, one 1)
# "1211" -> "111221" (one 1, one 2, two 1s)
#
# Key edge cases for prediction questions:
#   - Single digit: "5" -> "15"
#   - All same: "3333" -> "43"
#   - Empty string: "" -> ""
#   - Long alternating: "121212" -> "111211121112" 
#     (one 1, one 2, one 1, one 2, one 1, one 2)
#   - Non-digit characters: not supported (undefined behavior)
#   - The output format is always count then digit: "15" means "one 5"
# ============================================================

def look_and_say(s: str) -> str:
    if not s:
        return ""

    result = []
    current_char = s[0]
    count = 1

    for i in range(1, len(s)):
        if s[i] == current_char:
            count += 1
        else:
            result.append(str(count) + current_char)
            current_char = s[i]
            count = 1

    result.append(str(count) + current_char)
    return "".join(result)


# ---------- Test suite ----------
assert look_and_say("") == ""
assert look_and_say("1") == "11"
assert look_and_say("11") == "21"
assert look_and_say("21") == "1211"
assert look_and_say("1211") == "111221"
assert look_and_say("111221") == "312211"
assert look_and_say("5") == "15"
assert look_and_say("3333") == "43"
assert look_and_say("1222111") == "113231"
# '1' x1 -> "11", '2' x3 -> "32", '1' x3 -> "31" = "113231"
assert look_and_say("9") == "19"
assert look_and_say("aabb") == "2a2b"  # Works on any characters
print("✓ look_and_say: all tests passed")


# ============================================================
# SUMMARY: Suggested prediction questions per function
# ============================================================
# These are the calls you'd show participants. Mix easy ones
# (answerable even with a vague spec) and hard ones (require
# detailed spec knowledge to get right).
#
# RUN-LENGTH ENCODE:
#   Easy:  fn("aaabbb")        -> "a3b3"
#   Easy:  fn("")              -> ""
#   Medium: fn("abc")          -> "a1b1c1"  (do singles get counts?)
#   Hard:  fn("aAa")           -> "a1A1a1"  (case sensitivity)
#   Hard:  fn("   ")           -> " 3"      (spaces as characters)
#   Hard:  fn("112")           -> "1221"    (digits in input: ambiguous output)
#
# VERSION COMPARISON:
#   Easy:  fn("1.2", "1.3")        -> -1
#   Easy:  fn("2.0", "2.0")        -> 0
#   Medium: fn("1.0", "1.0.0")     -> 0  (trailing zeros)
#   Hard:  fn("01.2", "1.2")       -> 0  (leading zeros parsed as int)
#   Hard:  fn("", "")              -> 0  (empty string handling)
#   Hard:  fn("1.2.3.4", "1.2.3.4.0.0") -> 0
#
# COLUMN TO LABEL:
#   Easy:  fn(1)    -> "A"
#   Easy:  fn(3)    -> "C"
#   Medium: fn(26)  -> "Z"   (not "AA" — this trips people up)
#   Hard:  fn(27)   -> "AA"  (the bijective transition)
#   Hard:  fn(52)   -> "AZ"
#   Hard:  fn(53)   -> "BA"
#   Hard:  fn(702)  -> "ZZ"
#   Hard:  fn(703)  -> "AAA"
#
# LOOK-AND-SAY:
#   Easy:  fn("1")      -> "11"
#   Easy:  fn("11")     -> "21"
#   Medium: fn("21")    -> "1211"
#   Medium: fn("5")     -> "15"  (not "51" — count comes first)
#   Hard:  fn("3333")   -> "43"  (four 3s)
#   Hard:  fn("1211")   -> "111221"
#   Hard:  fn("")       -> ""
# ============================================================

print("\n=== All reference implementations verified ===")
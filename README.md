This Python module provides a collection of string constants and utility classes for common text processing tasks.

## Features

- Predefined string constants for:
  - Whitespace characters
  - ASCII lowercase and uppercase letters
  - ASCII digits, hexadecimal and octal digits
  - ASCII punctuation and printable characters
- `capwords` function to capitalize each word in a string
- `Template` class for performing `$`-based string substitutions
- `Formatter` class for advanced string formatting (based on PEP 3101)

## Included Constants

- `whitespace`: All ASCII whitespace characters (`' \t\n\r\v\f'`)
- `ascii_lowercase`: `'abcdefghijklmnopqrstuvwxyz'`
- `ascii_uppercase`: `'ABCDEFGHIJKLMNOPQRSTUVWXYZ'`
- `ascii_letters`: Combination of lowercase and uppercase ASCII letters
- `digits`: `'0123456789'`
- `hexdigits`: `'0123456789abcdefABCDEF'`
- `octdigits`: `'01234567'`
- `punctuation`: Standard ASCII punctuation symbols
- `printable`: All printable ASCII characters (includes letters, digits, punctuation, and whitespace)

## Functions

### `capwords(s, sep=None)`
Capitalizes each word in a string. If `sep` is provided, it's used to split and join the words. Otherwise, runs of whitespace are replaced with a single space.

## Classes

### `Template`
A class supporting `$`-based string substitution similar to shell scripting. Includes methods like:

- `substitute()`: Replaces all placeholders with provided values.
- `safe_substitute()`: Same as `substitute` but leaves placeholders intact if values are missing.
- `get_identifiers()`: Returns a list of all placeholders found in the template.
- `is_valid()`: Checks if the template has any invalid placeholders.

### `Formatter`
Supports complex string formatting. Compatible with Python’s format string syntax. Methods include:

- `format()`: Formats the given string with positional and keyword arguments.
- `vformat()`, `get_field()`, `convert_field()`, and others for internal formatting logic.

## Usage Example

```python
from your_module import ascii_letters, capwords, Template

print(ascii_letters)  # Output: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ

text = "hello world"
print(capwords(text))  # Output: Hello World

t = Template("Hello, $name!")
print(t.substitute(name="Alice"))  # Output: Hello, Alice!

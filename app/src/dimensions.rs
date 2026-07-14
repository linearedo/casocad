//! Dimension/measurement entry parsing — the port of `app/dimensions.py`.
//!
//! Text like `"2km"`, `"1 kilometre + 500m"`, `"10 x 20"`, `"(1m+5)/2"` is
//! parsed into values expressed in the current working unit (`unit_factor`
//! meters per unit). Explicit units win; bare numbers read in the working
//! unit. Expressions support `+ - * /` and parentheses; multiple values are
//! separated by `x`, `;` or `,` (or whitespace in the plain-number fallback).

use crate::state::LengthUnit;

/// Display counterpart of the entry parser: a length in meters rendered in
/// the working unit, trimmed to at most 4 decimals (`2.5 m`, `120 mm`).
pub fn format_length(meters: f64, unit: &LengthUnit) -> String {
    let shown = meters / unit.factor;
    let mut text = format!("{shown:.4}");
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    format!("{} {}", text, unit.key)
}

/// (spelling, meters per unit); longest spellings first so greedy unit
/// matching picks "mm" before "m".
const UNIT_SPELLINGS: [(&str, f64); 29] = [
    ("kilometers", 1000.0),
    ("kilometres", 1000.0),
    ("kilometer", 1000.0),
    ("kilometre", 1000.0),
    ("millimeters", 0.001),
    ("millimetres", 0.001),
    ("millimeter", 0.001),
    ("millimetre", 0.001),
    ("centimeters", 0.01),
    ("centimetres", 0.01),
    ("centimeter", 0.01),
    ("centimetre", 0.01),
    ("meters", 1.0),
    ("metres", 1.0),
    ("meter", 1.0),
    ("metre", 1.0),
    ("inches", 0.0254),
    ("inch", 0.0254),
    ("feet", 0.3048),
    ("foot", 0.3048),
    ("km", 1000.0),
    ("mm", 0.001),
    ("cm", 0.01),
    ("in", 0.0254),
    ("ft", 0.3048),
    ("m", 1.0),
    ("\"", 0.0254),
    ("'", 0.3048),
    ("", 0.0),
];

fn match_unit(text: &str) -> Option<(usize, f64)> {
    let lower = text.to_ascii_lowercase();
    for (spelling, factor) in UNIT_SPELLINGS {
        if !spelling.is_empty() && lower.starts_with(spelling) {
            // A unit word must not continue into more letters ("min" is not
            // "m"+"in"); symbol units (" and ') have no such issue.
            let tail = &text[spelling.len()..];
            if spelling.chars().all(|c| c.is_ascii_alphabetic())
                && tail.starts_with(|c: char| c.is_ascii_alphabetic())
            {
                continue;
            }
            return Some((spelling.len(), factor));
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Token {
    /// A numeric literal already converted to working units.
    Value(f64),
    Plus,
    Minus,
    Star,
    Slash,
    Open,
    Close,
}

/// Meters per typed literal: explicit units win, bare numbers are read in
/// the working unit — mirrors `_literal_factor`.
fn literal_factor(unit: Option<f64>, unit_factor: f64) -> f64 {
    unit.unwrap_or(unit_factor)
}

fn scan_number(text: &str) -> Option<(usize, f64)> {
    let bytes = text.as_bytes();
    let mut index = 0;
    if index < bytes.len() && (bytes[index] == b'+' || bytes[index] == b'-') {
        index += 1;
    }
    let digits_start = index;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    let integer_digits = index - digits_start;
    let mut fraction_digits = 0;
    if index < bytes.len() && bytes[index] == b'.' {
        index += 1;
        let start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        fraction_digits = index - start;
    }
    if integer_digits == 0 && fraction_digits == 0 {
        return None;
    }
    // Optional exponent.
    if index < bytes.len() && (bytes[index] == b'e' || bytes[index] == b'E') {
        let mut cursor = index + 1;
        if cursor < bytes.len() && (bytes[cursor] == b'+' || bytes[cursor] == b'-') {
            cursor += 1;
        }
        let start = cursor;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            cursor += 1;
        }
        if cursor > start {
            index = cursor;
        }
    }
    text[..index].parse::<f64>().ok().map(|value| (index, value))
}

/// Tokenize one expression: numbers (with optional trailing unit, spaces
/// allowed between) become working-unit `Value`s; `+ - * / ( )` pass through.
fn tokenize(text: &str, unit_factor: f64) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let trimmed = rest.trim_start();
        if trimmed.is_empty() {
            break;
        }
        rest = trimmed;
        let first = rest.chars().next().expect("non-empty");
        // A sign is part of the number only where Python's literal replacer
        // keeps it: after an operator/open-paren or at the start. Our token
        // stream expresses that as "previous token is not a Value/Close".
        let sign_can_start_number = !matches!(tokens.last(), Some(Token::Value(_) | Token::Close));
        let number = if first.is_ascii_digit()
            || first == '.'
            || ((first == '+' || first == '-') && sign_can_start_number)
        {
            scan_number(rest)
        } else {
            None
        };
        if let Some((consumed, value)) = number {
            rest = &rest[consumed..];
            // Optional unit after whitespace.
            let after_spaces = rest.trim_start();
            let mut unit = None;
            if let Some((unit_len, factor)) = match_unit(after_spaces) {
                unit = Some(factor);
                let spaces = rest.len() - after_spaces.len();
                rest = &rest[spaces + unit_len..];
            }
            tokens.push(Token::Value(
                value * literal_factor(unit, unit_factor) / unit_factor,
            ));
            continue;
        }
        let token = match first {
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '(' => Token::Open,
            ')' => Token::Close,
            other => return Err(format!("unexpected character: {other}")),
        };
        tokens.push(token);
        rest = &rest[first.len_utf8()..];
    }
    if tokens.is_empty() {
        return Err("dimension entries must contain at least one number".to_string());
    }
    Ok(tokens)
}

struct ExpressionParser<'tokens> {
    tokens: &'tokens [Token],
    position: usize,
}

impl ExpressionParser<'_> {
    fn peek(&self) -> Option<Token> {
        self.tokens.get(self.position).copied()
    }

    fn advance(&mut self) -> Option<Token> {
        let token = self.peek();
        if token.is_some() {
            self.position += 1;
        }
        token
    }

    fn expression(&mut self) -> Result<f64, String> {
        let mut value = self.term()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.advance();
                    value += self.term()?;
                }
                Some(Token::Minus) => {
                    self.advance();
                    value -= self.term()?;
                }
                _ => return Ok(value),
            }
        }
    }

    fn term(&mut self) -> Result<f64, String> {
        let mut value = self.unary()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.advance();
                    value *= self.unary()?;
                }
                Some(Token::Slash) => {
                    self.advance();
                    let divisor = self.unary()?;
                    value /= divisor;
                }
                _ => return Ok(value),
            }
        }
    }

    fn unary(&mut self) -> Result<f64, String> {
        match self.advance() {
            Some(Token::Plus) => self.unary(),
            Some(Token::Minus) => Ok(-self.unary()?),
            Some(Token::Value(value)) => Ok(value),
            Some(Token::Open) => {
                let value = self.expression()?;
                match self.advance() {
                    Some(Token::Close) => Ok(value),
                    _ => Err("unbalanced parentheses".to_string()),
                }
            }
            _ => Err("invalid dimension expression".to_string()),
        }
    }
}

fn parse_expression(text: &str, unit_factor: f64) -> Result<f64, String> {
    let tokens = tokenize(text, unit_factor)?;
    let mut parser = ExpressionParser {
        tokens: &tokens,
        position: 0,
    };
    let value = parser.expression()?;
    if parser.position != tokens.len() {
        return Err("invalid dimension expression".to_string());
    }
    if !value.is_finite() {
        return Err("dimension expression must be finite".to_string());
    }
    Ok(value)
}

/// Fallback for whitespace-separated plain entries like `"10 20"` or
/// `"1m 2m"`: independent number(+unit) tokens, separators limited to
/// whitespace and `,;xX` (mirrors the Python regex-scan branch).
fn parse_plain_list(text: &str, unit_factor: f64) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let trimmed =
            rest.trim_start_matches(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | 'x' | 'X'));
        if trimmed.is_empty() {
            break;
        }
        rest = trimmed;
        let Some((consumed, value)) = scan_number(rest) else {
            return Err("dimension entries must be numbers with optional units".to_string());
        };
        rest = &rest[consumed..];
        let after_spaces = rest.trim_start();
        let mut unit = None;
        if let Some((unit_len, factor)) = match_unit(after_spaces) {
            unit = Some(factor);
            let spaces = rest.len() - after_spaces.len();
            rest = &rest[spaces + unit_len..];
        }
        values.push(value * literal_factor(unit, unit_factor) / unit_factor);
    }
    if values.is_empty() {
        return Err("dimension entries must contain at least one number".to_string());
    }
    Ok(values)
}

/// Parse dimension text into values expressed in the working unit
/// (`unit_factor` meters per unit; 1.0 keeps meters in, meters out).
pub fn parse_measurement_entry(text: &str, unit_factor: f64) -> Result<Vec<f64>, String> {
    let parts: Vec<&str> = text
        .split([',', ';', 'x', 'X'])
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if !parts.is_empty() {
        let expression_values: Result<Vec<f64>, String> = parts
            .iter()
            .map(|part| parse_expression(part, unit_factor))
            .collect();
        if let Ok(values) = expression_values {
            return Ok(values);
        }
    }
    parse_plain_list(text, unit_factor)
}

/// Positive finite values only (sizes).
pub fn parse_dimension_entry(text: &str, unit_factor: f64) -> Result<Vec<f64>, String> {
    let values = parse_measurement_entry(text, unit_factor)?;
    if values
        .iter()
        .any(|value| *value <= 0.0 || !value.is_finite())
    {
        return Err("dimension entries must be positive finite numbers".to_string());
    }
    Ok(values)
}

/// Finite values, sign allowed (displacements — reserved for typed move
/// deltas, mirrors the Python API).
#[allow(dead_code)]
pub fn parse_displacement_entry(text: &str, unit_factor: f64) -> Result<Vec<f64>, String> {
    let values = parse_measurement_entry(text, unit_factor)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err("displacement entries must be finite numbers".to_string());
    }
    Ok(values)
}

/// A unitless arithmetic expression (angles, factors).
pub fn parse_scalar_entry(text: &str) -> Result<f64, String> {
    let tokens = tokenize(text, 1.0)?;
    // Scalar entries reject units: every literal must have been bare, which
    // is equivalent to re-tokenizing with unit spellings rejected. Simpler:
    // Python's parse goes through `ast.parse`, so any unit text errors out.
    if text
        .chars()
        .any(|c| c.is_ascii_alphabetic() || c == '"' || c == '\'')
    {
        return Err("invalid scalar expression".to_string());
    }
    let mut parser = ExpressionParser {
        tokens: &tokens,
        position: 0,
    };
    let value = parser.expression()?;
    if parser.position != tokens.len() || !value.is_finite() {
        return Err("invalid scalar expression".to_string());
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ports of tests/test_dimensions.py.
    #[test]
    fn accepts_kilometers() {
        assert_eq!(parse_measurement_entry("2km", 1.0).unwrap(), vec![2000.0]);
        assert_eq!(
            parse_measurement_entry("1 kilometre + 500m", 1.0).unwrap(),
            vec![1500.0]
        );
    }

    #[test]
    fn converts_explicit_units_to_working_unit() {
        assert_eq!(parse_dimension_entry("0.001m", 0.001).unwrap(), vec![1.0]);
        assert_eq!(parse_dimension_entry("10mm", 0.001).unwrap(), vec![10.0]);
        assert_eq!(parse_dimension_entry("1000m", 1000.0).unwrap(), vec![1.0]);
    }

    #[test]
    fn reads_bare_numbers_in_working_unit() {
        assert_eq!(parse_dimension_entry("10", 0.001).unwrap(), vec![10.0]);
        assert_eq!(
            parse_dimension_entry("10 x 20", 0.01).unwrap(),
            vec![10.0, 20.0]
        );
        assert_eq!(parse_dimension_entry("1m + 5", 0.001).unwrap(), vec![1005.0]);
    }

    #[test]
    fn unit_registry_is_parseable_and_consistent() {
        let keys: Vec<&str> = crate::state::LENGTH_UNITS.iter().map(|unit| unit.key).collect();
        assert_eq!(keys, vec!["km", "m", "cm", "mm"]);
        for unit in crate::state::LENGTH_UNITS {
            assert_eq!(
                parse_measurement_entry(&format!("1{}", unit.key), 1.0).unwrap(),
                vec![unit.factor]
            );
        }
    }

    #[test]
    fn whitespace_fallback_and_expressions() {
        assert_eq!(
            parse_measurement_entry("10 20", 1.0).unwrap(),
            vec![10.0, 20.0]
        );
        assert_eq!(
            parse_measurement_entry("(1m+5)/2", 0.001).unwrap(),
            vec![502.5]
        );
        assert_eq!(
            parse_displacement_entry("-5mm; 2cm", 0.001).unwrap(),
            vec![-5.0, 20.0]
        );
        assert!(parse_dimension_entry("-5", 1.0).is_err());
        assert!(parse_measurement_entry("nonsense", 1.0).is_err());
    }

    #[test]
    fn format_length_trims_and_converts() {
        let meters = crate::state::LENGTH_UNITS[1];
        let millimeters = crate::state::LENGTH_UNITS[3];
        assert_eq!(format_length(2.5, &meters), "2.5 m");
        assert_eq!(format_length(2.0, &meters), "2 m");
        assert_eq!(format_length(0.12, &millimeters), "120 mm");
        assert_eq!(format_length(0.123456789, &meters), "0.1235 m");
        assert_eq!(format_length(0.0, &meters), "0 m");
    }

    #[test]
    fn scalar_entry_rejects_units() {
        assert_eq!(parse_scalar_entry("2*3 + 1").unwrap(), 7.0);
        assert!(parse_scalar_entry("2m").is_err());
    }
}

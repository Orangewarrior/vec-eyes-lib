use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClassificationLabel {
    Spam,
    Malware,
    Phishing,
    Anomaly,
    Fuzzing,
    WebAttack,
    Flood,
    Porn,
    RawData,
    BlockList,
    Virus,
    Human,
    Animal,
    Cancer,
    Fungus,
    Bacteria,
    Free,
    Custom(String),
}

impl ClassificationLabel {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Spam => "SPAM",
            Self::Malware => "MALWARE",
            Self::Phishing => "PHISHING",
            Self::Anomaly => "ANOMALY",
            Self::Fuzzing => "FUZZING",
            Self::WebAttack => "WEB_ATTACK",
            Self::Flood => "FLOOD",
            Self::Porn => "PORN",
            Self::RawData => "RAW_DATA",
            Self::BlockList => "BLOCK_LIST",
            Self::Virus => "VIRUS",
            Self::Human => "HUMAN",
            Self::Animal => "ANIMAL",
            Self::Cancer => "CANCER",
            Self::Fungus => "FUNGUS",
            Self::Bacteria => "BACTERIA",
            Self::Free => "FREE",
            Self::Custom(value) => value.as_str(),
        }
    }
}

impl Display for ClassificationLabel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ClassificationLabel {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err("label cannot be empty");
        }
        if s.len() > 64 {
            return Err("label exceeds maximum length of 64 characters");
        }
        if s.chars().any(|c| c.is_control()) {
            return Err("label contains control characters");
        }
        Ok(match s.to_ascii_uppercase().as_str() {
            "SPAM" => Self::Spam,
            "MALWARE" => Self::Malware,
            "PHISHING" => Self::Phishing,
            "ANOMALY" => Self::Anomaly,
            "FUZZING" => Self::Fuzzing,
            "WEB_ATTACK" => Self::WebAttack,
            "FLOOD" => Self::Flood,
            "PORN" => Self::Porn,
            "RAW_DATA" => Self::RawData,
            "BLOCK_LIST" => Self::BlockList,
            "VIRUS" => Self::Virus,
            "HUMAN" => Self::Human,
            "ANIMAL" => Self::Animal,
            "CANCER" => Self::Cancer,
            "FUNGUS" => Self::Fungus,
            "BACTERIA" => Self::Bacteria,
            "FREE" => Self::Free,
            _ => Self::Custom(s.to_string()),
        })
    }
}

impl Serialize for ClassificationLabel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ClassificationLabel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        ClassificationLabel::from_str(&value).map_err(|_| de::Error::custom("invalid label"))
    }
}

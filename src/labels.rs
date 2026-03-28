use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
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
}

impl ClassificationLabel {
    pub fn as_str(&self) -> &'static str {
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
        }
    }
}

impl Display for ClassificationLabel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ClassificationLabel {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "SPAM" => Ok(Self::Spam),
            "MALWARE" => Ok(Self::Malware),
            "PHISHING" => Ok(Self::Phishing),
            "ANOMALY" => Ok(Self::Anomaly),
            "FUZZING" => Ok(Self::Fuzzing),
            "WEB_ATTACK" => Ok(Self::WebAttack),
            "FLOOD" => Ok(Self::Flood),
            "PORN" => Ok(Self::Porn),
            "RAW_DATA" => Ok(Self::RawData),
            "BLOCK_LIST" => Ok(Self::BlockList),
            "VIRUS" => Ok(Self::Virus),
            "HUMAN" => Ok(Self::Human),
            "ANIMAL" => Ok(Self::Animal),
            "CANCER" => Ok(Self::Cancer),
            "FUNGUS" => Ok(Self::Fungus),
            "BACTERIA" => Ok(Self::Bacteria),
            "FREE" => Ok(Self::Free),
            _ => Err(()),
        }
    }
}

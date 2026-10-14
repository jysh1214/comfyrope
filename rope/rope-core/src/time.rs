/// The local time to the second, with its UTC offset: 2026-10-14T10:19:31+08:00.
pub fn now() -> String {
    chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
}

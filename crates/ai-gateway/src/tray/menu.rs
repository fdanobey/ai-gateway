use super::ServerStatus;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrayMenuAction {
    OpenDashboard,
    ViewServerStatus,
    Quit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrayMenuItem {
    pub action: TrayMenuAction,
    pub label: String,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct TrayMenu {
    items: Vec<TrayMenuItem>,
}

impl TrayMenu {
    pub fn new(status: &ServerStatus) -> Self {
        Self {
            items: vec![
                TrayMenuItem {
                    action: TrayMenuAction::OpenDashboard,
                    label: "Open Admin Dashboard".to_string(),
                    enabled: true,
                },
                TrayMenuItem {
                    action: TrayMenuAction::ViewServerStatus,
                    label: Self::status_label(status),
                    enabled: false,
                },
                TrayMenuItem {
                    action: TrayMenuAction::Quit,
                    label: "Quit".to_string(),
                    enabled: true,
                },
            ],
        }
    }

    pub fn update_status(&mut self, status: &ServerStatus) {
        if let Some(item) = self
            .items
            .iter_mut()
            .find(|item| item.action == TrayMenuAction::ViewServerStatus)
        {
            item.label = Self::status_label(status);
        }
    }

    pub fn items(&self) -> &[TrayMenuItem] {
        &self.items
    }

    fn status_label(status: &ServerStatus) -> String {
        if status.is_running {
            format!("Server Running: {}", status.address)
        } else {
            format!("Server Stopped: {}", status.address)
        }
    }
}

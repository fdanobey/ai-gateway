use std::path::PathBuf;
use std::time::Duration;

use image::GenericImageView;

use super::TrayError;

#[derive(Debug, Clone)]
pub struct SplashScreen {
    logo_path: PathBuf,
    duration: Duration,
    visible: bool,
    dimensions: Option<(u32, u32)>,
}

impl SplashScreen {
    pub fn new(logo_path: impl Into<PathBuf>, duration: Duration) -> Self {
        Self {
            logo_path: logo_path.into(),
            duration,
            visible: false,
            dimensions: None,
        }
    }

    pub async fn show(&mut self) -> Result<(), TrayError> {
        let image = image::open(&self.logo_path)?;
        self.dimensions = Some(image.dimensions());
        self.visible = true;
        tracing::info!(logo = %self.logo_path.display(), "Showing first-launch splash screen");
        tokio::time::sleep(self.duration / 2).await;
        Ok(())
    }

    pub async fn animate_to_tray(&self) {
        tracing::info!(logo = %self.logo_path.display(), "Animating splash screen toward tray area");
        tokio::time::sleep(self.duration / 2).await;
    }

    pub fn close(&mut self) {
        self.visible = false;
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.dimensions
    }
}

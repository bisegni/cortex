use anyhow::Result;
use cortex_core::Signal;
use crossterm::{event::{self,Event,KeyCode},execute,terminal::{disable_raw_mode,enable_raw_mode,EnterAlternateScreen,LeaveAlternateScreen}};
use futures::StreamExt;
use ratatui::{backend::CrosstermBackend,layout::{Constraint,Direction,Layout},widgets::{Block,Borders,Paragraph},Terminal};
use std::{collections::VecDeque,io,time::Duration};

#[tokio::main]
async fn main()->Result<()> {
    let nc=async_nats::connect(std::env::var("NATS_URL").unwrap_or_else(|_|"127.0.0.1:4222".into())).await?;
    let mut sub=nc.subscribe("cortex.*.signal").await?;
    enable_raw_mode()?; let mut out=io::stdout(); execute!(out,EnterAlternateScreen)?;
    let mut term=Terminal::new(CrosstermBackend::new(out))?;
    let mut log:VecDeque<Signal>=VecDeque::new();
    loop {
        while let Ok(Some(msg))=tokio::time::timeout(Duration::from_millis(1),sub.next()).await {
            if let Ok(s)=serde_json::from_slice::<Signal>(&msg.payload){ log.push_front(s); if log.len()>40{log.pop_back();} }
        }
        term.draw(|f|{
            let rows=Layout::default().direction(Direction::Vertical).constraints([Constraint::Length(8),Constraint::Min(5),Constraint::Length(2)]).split(f.area());
            let activity=log.iter().take(12).map(|s|format!("{:<12} {:<26} {:.2}",s.source,s.symbol,s.activation)).collect::<Vec<_>>().join("\n");
            let stream=log.iter().take(20).map(|s|format!("{:>8}  {:<11} -> {:<11}  {}",s.timestamp_ms%100000,s.source,s.kind,s.symbol)).collect::<Vec<_>>().join("\n");
            f.render_widget(Paragraph::new(activity).block(Block::default().title(" Cortical Activity / Workspace ").borders(Borders::ALL)),rows[0]);
            f.render_widget(Paragraph::new(stream).block(Block::default().title(" Live Neural Signal Bus ").borders(Borders::ALL)),rows[1]);
            f.render_widget(Paragraph::new(" q: quit   |   each cortex is an independent process; TUI only observes NATS"),rows[2]);
        })?;
        if event::poll(Duration::from_millis(30))? { if let Event::Key(k)=event::read()? { if k.code==KeyCode::Char('q'){break;} } }
    }
    disable_raw_mode()?; execute!(term.backend_mut(),LeaveAlternateScreen)?; term.show_cursor()?;
    Ok(())
}

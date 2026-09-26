use anyhow::Result;
use cortex_core::{Signal,CONNECTOME};
use crossterm::{event::{self,Event,KeyCode},execute,terminal::{disable_raw_mode,enable_raw_mode,EnterAlternateScreen,LeaveAlternateScreen}};
use futures::StreamExt;
use ratatui::{backend::CrosstermBackend,layout::{Constraint,Direction,Layout},style::{Color,Modifier,Style},text::{Line,Span},widgets::{Block,Borders,Paragraph},Terminal};
use std::{collections::{HashMap,VecDeque},io,time::{Duration,Instant}};

struct EdgeActivity{events:VecDeque<Instant>,last_weight:f32}
fn heat(rate:f32)->Color{if rate>15.0{Color::LightRed}else if rate>7.0{Color::LightYellow}else if rate>2.0{Color::LightGreen}else if rate>0.0{Color::Cyan}else{Color::DarkGray}}
fn bar(rate:f32)->String{let n=((rate/2.0).ceil()as usize).min(12);format!("{}{}","█".repeat(n),"·".repeat(12-n))}
#[tokio::main]
async fn main()->Result<()>{
 let nc=async_nats::connect(std::env::var("NATS_URL").unwrap_or_else(|_|"127.0.0.1:4222".into())).await?;
 let mut sub=nc.subscribe("cortex.>").await?;enable_raw_mode()?;let mut out=io::stdout();execute!(out,EnterAlternateScreen)?;
 let mut term=Terminal::new(CrosstermBackend::new(out))?;let mut edges:HashMap<(String,String),EdgeActivity>=HashMap::new();let mut log:VecDeque<Signal>=VecDeque::new();
 loop{
  while let Ok(Some(msg))=tokio::time::timeout(Duration::from_millis(1),sub.next()).await{if let Ok(s)=serde_json::from_slice::<Signal>(&msg.payload){
   let e=edges.entry((s.source.clone(),s.target.clone())).or_insert(EdgeActivity{events:VecDeque::new(),last_weight:s.connection_weight});e.events.push_back(Instant::now());e.last_weight=s.connection_weight;
   log.push_front(s);if log.len()>12{log.pop_back();}
  }}
  let now=Instant::now();for e in edges.values_mut(){while e.events.front().is_some_and(|t|now.duration_since(*t)>Duration::from_secs(3)){e.events.pop_front();}}
  term.draw(|f|{
   let rows=Layout::default().direction(Direction::Vertical).constraints([Constraint::Length(11),Constraint::Min(10),Constraint::Length(5),Constraint::Length(1)]).split(f.area());
   let brain=vec![
    Line::from(vec![Span::styled("                         ┌──────────────┐",Style::default().fg(Color::DarkGray))]),
    Line::from(vec![Span::raw("  CAMERA ─────► "),Span::styled("VISUAL",Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD)),Span::raw(" ─────► "),Span::styled("PERCEPTUAL",Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD)),Span::raw(" ◄─────┐")]),
    Line::from("                                      │        │"),
    Line::from(vec![Span::raw("                         ┌────────────┼────────┐")]),
    Line::from(vec![Span::raw("                         ▼            ▼        │")]),
    Line::from(vec![Span::raw("                      "),Span::styled("MEMORY",Style::default().fg(Color::LightMagenta).add_modifier(Modifier::BOLD)),Span::raw(" ◄────► "),Span::styled("ASSOCIATIVE",Style::default().fg(Color::LightYellow).add_modifier(Modifier::BOLD)),Span::raw(" ─┘")]),
    Line::from(vec![Span::raw("                         ▲             ▲  │")]),
    Line::from(vec![Span::raw("                         │             │  ▼")]),
    Line::from(vec![Span::raw("                      "),Span::styled("WORKSPACE",Style::default().fg(Color::LightRed).add_modifier(Modifier::BOLD)),Span::raw("       "),Span::styled("LANGUAGE",Style::default().fg(Color::LightBlue).add_modifier(Modifier::BOLD))]),
   ];
   f.render_widget(Paragraph::new(brain).block(Block::default().title(" CORTEX — sparse connectome ").borders(Borders::ALL)),rows[0]);
   let mut lines=vec![Line::from(vec![Span::styled(" projection                    weight   traffic/3s    activity",Style::default().add_modifier(Modifier::BOLD))])];
   for c in CONNECTOME{let key=(c.source.to_string(),c.target.to_string());let e=edges.get(&key);let rate=e.map(|x|x.events.len()as f32/3.0).unwrap_or(0.0);let weight=e.map(|x|x.last_weight).unwrap_or(c.weight);let color=heat(rate);
    lines.push(Line::from(vec![Span::styled(format!(" {:<11} → {:<11}",c.source,c.target),Style::default().fg(color)),Span::raw(format!("   {:>4.2}     {:>6.1}/s    ",weight,rate)),Span::styled(bar(rate),Style::default().fg(color).add_modifier(if rate>7.0{Modifier::BOLD}else{Modifier::empty()}))]));
   }
   f.render_widget(Paragraph::new(lines).block(Block::default().title(" Connections — brighter = more recent traffic ").borders(Borders::ALL)),rows[1]);
   let stream=log.iter().take(3).map(|s|Line::from(vec![Span::styled(format!("{:<11}",s.source),Style::default().fg(Color::Cyan)),Span::raw(" → "),Span::styled(format!("{:<11}",s.target),Style::default().fg(Color::LightGreen)),Span::raw(format!(" {:<14} {}  a={:.2}",s.kind,s.symbol,s.activation))])).collect::<Vec<_>>();
   f.render_widget(Paragraph::new(stream).block(Block::default().title(" Latest signals ").borders(Borders::ALL)),rows[2]);
   f.render_widget(Paragraph::new(" q quit   |   weight = connection strength   |   traffic = observed messages/sec over last 3s"),rows[3]);
  })?;
  if event::poll(Duration::from_millis(40))?{if let Event::Key(k)=event::read()?{if k.code==KeyCode::Char('q'){break}}}
 }
 disable_raw_mode()?;execute!(term.backend_mut(),LeaveAlternateScreen)?;term.show_cursor()?;Ok(())
}

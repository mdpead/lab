import ast
import math
import re
from collections.abc import Callable, Iterable

# The README's charts, as plain SVG read from the stage logs under models/:
#   docs/sft_curves.svg  SFT validation loss for the three output-prediction runs
#   docs/pipeline.svg    one panel per stage for the model behind the headline result
# No plotting library. The colours switch with the viewer's light or dark theme.

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

STYLE = f"""
  .surface {{ fill: #fcfcfb; }}
  .title {{ fill: #0b0b0b; font: 600 16px {FONT}; }}
  .panel-title {{ fill: #0b0b0b; font: 600 14px {FONT}; }}
  .subtitle, .label {{ fill: #52514e; font: 13px {FONT}; }}
  .value {{ fill: #0b0b0b; font: 600 13px {FONT}; }}
  .tick {{ fill: #898781; font: 12px {FONT}; font-variant-numeric: tabular-nums; }}
  .grid {{ stroke: #e1e0d9; stroke-width: 1; }}
  .axis {{ stroke: #c3c2b7; stroke-width: 1; }}
  .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  circle.ring {{ stroke: #fcfcfb; stroke-width: 2; }}
  .s1 {{ stroke: #2a78d6; }} .s1.dot {{ fill: #2a78d6; }}
  .s2 {{ stroke: #eb6834; }} .s2.dot {{ fill: #eb6834; }}
  .s3 {{ stroke: #1baf7a; }} .s3.dot {{ fill: #1baf7a; }}
  @media (prefers-color-scheme: dark) {{
    .surface {{ fill: #1a1a19; }}
    .title, .panel-title, .value {{ fill: #ffffff; }}
    .subtitle, .label {{ fill: #c3c2b7; }}
    .grid {{ stroke: #2c2c2a; }}
    .axis {{ stroke: #383835; }}
    circle.ring {{ stroke: #1a1a19; }}
    .s1 {{ stroke: #3987e5; }} .s1.dot {{ fill: #3987e5; }}
    .s2 {{ stroke: #d95926; }} .s2.dot {{ fill: #d95926; }}
    .s3 {{ stroke: #199e70; }} .s3.dot {{ fill: #199e70; }}
  }}
"""


def validation(run: str, stage: str, key: str) -> dict[int, float]:
    """{step: value} from a stage log, the last record per step where a resume repeated it."""
    values: dict[int, float] = {}
    with open(f"models/{run}/{stage}/{stage}.log") as f:
        for line in f:
            match = re.search(r"(\{'type': 'validation'.*\})\s*$", line)
            if match:
                record = ast.literal_eval(match.group(1))
                values[record["step_no"]] = record[key]
    return dict(sorted(values.items()))


class Panel:
    """A plot area at (x, y, w, h) with its own axes. `log` scales y logarithmically."""

    def __init__(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        x_max: float,
        y_min: float,
        y_max: float,
        log: bool = False,
    ) -> None:
        self.x, self.y, self.w, self.h = x, y, w, h
        self.x_max, self.y_min, self.y_max, self.log = x_max, y_min, y_max, log

    def px(self, step: float) -> float:
        return self.x + self.w * step / self.x_max

    def py(self, value: float) -> float:
        scale: Callable[[float], float] = math.log10 if self.log else (lambda v: v)
        span = scale(self.y_max) - scale(self.y_min)
        return self.y + self.h * (scale(self.y_max) - scale(value)) / span

    def axes(
        self,
        y_ticks: Iterable[float],
        y_format: Callable[[float], str],
        x_ticks: Iterable[int],
        x_label: str,
        y_label: str | None,
    ) -> list[str]:
        parts = []
        for value in y_ticks:
            y = self.py(value)
            parts.append(
                f'<line class="grid" x1="{self.x}" y1="{y:.1f}" '
                f'x2="{self.x + self.w}" y2="{y:.1f}"/>'
            )
            parts.append(
                f'<text class="tick" x="{self.x - 8}" y="{y + 4:.1f}" '
                f'text-anchor="end">{y_format(value)}</text>'
            )
        base = self.y + self.h
        parts.append(
            f'<line class="axis" x1="{self.x}" y1="{base}" x2="{self.x + self.w}" y2="{base}"/>'
        )
        for step in x_ticks:
            parts.append(
                f'<text class="tick" x="{self.px(step):.1f}" y="{base + 18}" '
                f'text-anchor="middle">{step:,}</text>'
            )
        parts.append(
            f'<text class="label" x="{self.x + self.w / 2:.1f}" y="{base + 40}" '
            f'text-anchor="middle">{x_label}</text>'
        )
        if y_label:
            parts.append(
                f'<text class="label" '
                f'transform="translate({self.x - 50} {self.y + self.h / 2:.1f}) rotate(-90)" '
                f'text-anchor="middle">{y_label}</text>'
            )
        return parts

    def series(
        self,
        values: dict[int, float],
        cls: str,
        name: str,
        value_format: Callable[[float], str],
        label: str | None = None,
    ) -> list[str]:
        """A line with an end dot, its final value beside the dot, then an optional label."""
        points = " ".join(f"{self.px(s):.1f},{self.py(v):.1f}" for s, v in values.items())
        last_step, last = list(values.items())[-1]
        x, y = self.px(last_step), self.py(last)
        parts = [
            f'<polyline class="line {cls}" points="{points}"/>',
            f'<circle class="dot ring {cls}" cx="{x:.1f}" cy="{y:.1f}" r="4">'
            f"<title>{name}: {value_format(last)} at step {last_step:,}</title></circle>",
            f'<text class="value" x="{x + 12:.1f}" y="{y + 4:.1f}">{value_format(last)}</text>',
        ]
        if label:
            parts.append(f'<text class="label" x="{x + 64:.1f}" y="{y + 4:.1f}">{label}</text>')
        return parts


def svg(width: int, height: int, title: str, desc: str, body: list[str]) -> str:
    return (
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
                f'width="{width}" height="{height}" role="img" aria-labelledby="title desc">',
                f'<title id="title">{title}</title>',
                f'<desc id="desc">{desc}</desc>',
                f"<style>{STYLE}</style>",
                f'<rect class="surface" width="{width}" height="{height}" rx="8"/>',
                *body,
                "</svg>",
            ]
        )
        + "\n"
    )


def sft_curves() -> str:
    runs = [
        ("base_lr3", "Pretrained, 3e-4", "s1"),
        ("sft_only", "From scratch, 3e-4", "s2"),
        ("base", "Pretrained, 1e-4", "s3"),
    ]
    panel = Panel(64, 92, 456, 276, 10000, 0.005, 0.4, log=True)
    body = [
        '<text class="title" x="20" y="30">SFT validation loss</text>',
        '<text class="subtitle" x="20" y="50">Lower is better. Same data and schedule; the '
        "runs differ in starting weights and learning rate.</text>",
    ]
    legend_x = panel.x
    for _, name, cls in runs:
        body.append(
            f'<line class="line {cls}" x1="{legend_x}" y1="72" x2="{legend_x + 18}" y2="72"/>'
        )
        body.append(f'<text class="label" x="{legend_x + 24}" y="76">{name}</text>')
        legend_x += 24 + 7 * len(name) + 24
    body += panel.axes(
        (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4),
        lambda v: f"{v:g}",
        range(0, 10001, 2000),
        "SFT step",
        "Validation loss, log scale",
    )
    for run, name, cls in runs:
        body += panel.series(
            validation(run, "sft", "loss_ce"), cls, name, lambda v: f"{v:.4f}", name
        )
    return svg(
        720,
        420,
        "SFT validation loss for three runs",
        "Validation loss over 10,000 SFT steps on a log scale. The pretrained run at 3e-4 is "
        "lowest throughout and ends at 0.0068, the from-scratch run at 3e-4 ends at 0.0101, "
        "and the pretrained run at 1e-4 levels off and ends at 0.0138.",
        body,
    )


def pipeline() -> str:
    pretrain = Panel(70, 116, 210, 178, 10000, 1.0, 4.0)
    sft = Panel(390, 116, 210, 178, 10000, 0.005, 0.2, log=True)
    rl = Panel(700, 116, 180, 178, 100, 0.80, 0.95)
    body = [
        '<text class="title" x="20" y="30">The full pipeline, stage by stage</text>',
        '<text class="subtitle" x="20" y="50">Pretraining from base; SFT and RL from base_lr3, '
        "which fine-tunes the same pretrained weights at 3e-4.</text>",
    ]
    panels = [
        (pretrain, "1. Pretrain", "Validation loss, held-out Python"),
        (sft, "2. SFT", "Validation loss, held-out traces"),
        (rl, "3. RL", "Greedy pass@1, 375 held-out prompts"),
    ]
    for panel, name, metric in panels:
        body.append(f'<text class="panel-title" x="{panel.x - 50}" y="80">{name}</text>')
        body.append(f'<text class="label" x="{panel.x - 50}" y="96">{metric}</text>')

    body += pretrain.axes((1, 2, 3, 4), lambda v: f"{v:g}", (0, 5000, 10000), "Pretrain step", None)
    body += pretrain.series(
        validation("base", "pretrain", "loss_ce"), "s1", "Pretrain", lambda v: f"{v:.3f}"
    )
    body += sft.axes(
        (0.005, 0.01, 0.02, 0.05, 0.1, 0.2), lambda v: f"{v:g}", (0, 5000, 10000), "SFT step", None
    )
    body += sft.series(validation("base_lr3", "sft", "loss_ce"), "s1", "SFT", lambda v: f"{v:.4f}")
    body += rl.axes((0.80, 0.85, 0.90, 0.95), lambda v: f"{v:.0%}", (0, 50, 100), "RL step", None)
    body += rl.series(validation("base_lr3", "rl", "pass_rate"), "s1", "RL", lambda v: f"{v:.1%}")
    return svg(
        960,
        360,
        "The full pipeline, stage by stage",
        "Three panels. Pretraining validation loss falls from 3.92 at step 500 to 1.465 at "
        "step 10,000. SFT validation loss, on a log scale, falls from 0.104 to 0.0068 over "
        "10,000 steps. RL greedy pass@1 stays between 88.5% and 90.9% over 100 steps, "
        "starting at 90.1% and ending at 90.9%.",
        body,
    )


def main() -> None:
    for path, content in (("docs/sft_curves.svg", sft_curves()), ("docs/pipeline.svg", pipeline())):
        with open(path, "w") as f:
            f.write(content)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

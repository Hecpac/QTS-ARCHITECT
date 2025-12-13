# SUPER AGENTE Auditor - Usage Examples

## Example 1: Basic Project Audit

### Scenario
You have a new project and want to perform an initial audit to understand its compliance status.

### Command
```bash
python3 super_agente_auditor.py
```

### Expected Output
```
================================================================================
SUPER AGENTE AUDITOR - Project Analysis
================================================================================

Analyzing project: /home/runner/work/QTS-ARCHITECT/QTS-ARCHITECT
Timestamp: 2025-12-13T04:51:51.123456

Running BMadMethodAnalyzer...
Running ICTPatternAnalyzer...
Running ArchitectureAnalyzer...
Running RiskManagementAnalyzer...
Running SecurityAnalyzer...
Running PerformanceAnalyzer...

================================================================================
AUDIT REPORT
================================================================================

Total Findings: 15

Findings by Severity:
  HIGH: 5
  MEDIUM: 7
  LOW: 3

...
```

## Example 2: Integration with Git Workflow

### Scenario
You want to audit your project before committing code changes.

### Pre-commit Hook Script
Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

echo "Running SUPER AGENTE Auditor..."
python3 super_agente_auditor.py --json-only -o /tmp/audit_report.json

# Check exit code
if [ $? -ne 0 ]; then
    echo "❌ Audit found CRITICAL issues!"
    echo "Please review /tmp/audit_report.json"
    exit 1
fi

# Check for critical findings in JSON
CRITICAL_COUNT=$(python3 -c "import json; data=json.load(open('/tmp/audit_report.json')); print(data['statistics']['by_severity']['critical'])")

if [ "$CRITICAL_COUNT" -gt 0 ]; then
    echo "❌ Found $CRITICAL_COUNT critical security issues!"
    echo "Please fix before committing."
    exit 1
fi

echo "✅ Audit passed!"
exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Example 3: CI/CD Pipeline Integration

### GitHub Actions Workflow
Create `.github/workflows/audit.yml`:

```yaml
name: SUPER AGENTE Audit

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  audit:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Run SUPER AGENTE Auditor
      run: |
        python3 super_agente_auditor.py --json-only -o audit_report.json
        cat audit_report.json | python3 -m json.tool
    
    - name: Check for Critical Issues
      run: |
        CRITICAL=$(python3 -c "import json; print(json.load(open('audit_report.json'))['statistics']['by_severity']['critical'])")
        if [ "$CRITICAL" -gt 0 ]; then
          echo "❌ Found $CRITICAL critical issues"
          exit 1
        fi
    
    - name: Upload Audit Report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: audit-report
        path: audit_report.json
    
    - name: Comment PR with Audit Results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = JSON.parse(fs.readFileSync('audit_report.json', 'utf8'));
          const stats = report.statistics;
          
          const comment = `## 🔍 SUPER AGENTE Audit Report
          
          **Total Findings:** ${stats.total_findings}
          
          **By Severity:**
          - 🔴 Critical: ${stats.by_severity.critical}
          - 🟠 High: ${stats.by_severity.high}
          - 🟡 Medium: ${stats.by_severity.medium}
          - 🟢 Low: ${stats.by_severity.low}
          - ℹ️ Info: ${stats.by_severity.info}
          
          See the audit report artifact for detailed findings.
          `;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

### GitLab CI Configuration
Create `.gitlab-ci.yml`:

```yaml
stages:
  - audit
  - test
  - deploy

audit_project:
  stage: audit
  image: python:3.9
  script:
    - python3 super_agente_auditor.py --json-only -o audit_report.json
    - python3 -m json.tool audit_report.json
    - |
      CRITICAL=$(python3 -c "import json; print(json.load(open('audit_report.json'))['statistics']['by_severity']['critical'])")
      if [ "$CRITICAL" -gt 0 ]; then
        echo "❌ Found $CRITICAL critical issues"
        exit 1
      fi
  artifacts:
    paths:
      - audit_report.json
    expire_in: 30 days
    reports:
      junit: audit_report.json
  allow_failure: false
```

## Example 4: Comparing Audits Over Time

### Scenario
Track improvements in your project by comparing audit reports over time.

### Script: `compare_audits.py`
```python
#!/usr/bin/env python3
import json
import sys
from datetime import datetime

def compare_audits(old_report, new_report):
    """Compare two audit reports"""
    with open(old_report) as f:
        old = json.load(f)
    with open(new_report) as f:
        new = json.load(f)
    
    print("=" * 80)
    print("AUDIT COMPARISON")
    print("=" * 80)
    print(f"\nOld Report: {old['timestamp']}")
    print(f"New Report: {new['timestamp']}")
    
    old_stats = old['statistics']
    new_stats = new['statistics']
    
    print(f"\n{'Category':<20} {'Old':<10} {'New':<10} {'Change':<10}")
    print("-" * 80)
    
    print(f"{'Total Findings':<20} {old_stats['total_findings']:<10} {new_stats['total_findings']:<10} {new_stats['total_findings'] - old_stats['total_findings']:+d}")
    
    print("\nBy Severity:")
    for severity in ['critical', 'high', 'medium', 'low', 'info']:
        old_count = old_stats['by_severity'][severity]
        new_count = new_stats['by_severity'][severity]
        change = new_count - old_count
        status = "✅" if change <= 0 else "❌"
        print(f"  {severity.capitalize():<17} {old_count:<10} {new_count:<10} {change:+d} {status}")
    
    # Calculate improvement score
    old_score = (old_stats['by_severity']['critical'] * 10 + 
                 old_stats['by_severity']['high'] * 5 +
                 old_stats['by_severity']['medium'] * 2 +
                 old_stats['by_severity']['low'] * 1)
    
    new_score = (new_stats['by_severity']['critical'] * 10 + 
                 new_stats['by_severity']['high'] * 5 +
                 new_stats['by_severity']['medium'] * 2 +
                 new_stats['by_severity']['low'] * 1)
    
    improvement = old_score - new_score
    print(f"\nWeighted Score Change: {improvement:+d}")
    
    if improvement > 0:
        print("✅ Project has improved!")
    elif improvement < 0:
        print("❌ Project has regressed!")
    else:
        print("➖ No change in overall score")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 compare_audits.py <old_report.json> <new_report.json>")
        sys.exit(1)
    
    compare_audits(sys.argv[1], sys.argv[2])
```

Usage:
```bash
# First audit
python3 super_agente_auditor.py -o audit_2025_01_01.json

# Make improvements...

# Second audit
python3 super_agente_auditor.py -o audit_2025_02_01.json

# Compare
python3 compare_audits.py audit_2025_01_01.json audit_2025_02_01.json
```

## Example 5: Selective Audit (Custom Script)

### Scenario
You only want to run specific analyzers.

### Script: `selective_audit.py`
```python
#!/usr/bin/env python3
from super_agente_auditor import (
    SuperAgenteAuditor, 
    SecurityAnalyzer, 
    PerformanceAnalyzer
)
from pathlib import Path

# Create auditor but use only specific analyzers
project_path = "."
auditor = SuperAgenteAuditor(project_path)

# Replace analyzers with only the ones you want
auditor.analyzers = [
    SecurityAnalyzer(project_path),
    PerformanceAnalyzer(project_path)
]

# Run audit
auditor.run_audit()
auditor.print_report()
auditor.save_report("selective_audit.json")
```

Usage:
```bash
python3 selective_audit.py
```

## Example 6: Automated Fix Suggestions

### Scenario
Generate a checklist of items to fix based on audit findings.

### Script: `generate_todo.py`
```python
#!/usr/bin/env python3
import json
import sys

def generate_todo(audit_report):
    """Generate TODO list from audit report"""
    with open(audit_report) as f:
        report = json.load(f)
    
    print("# TODO: Address Audit Findings\n")
    print(f"Generated from audit: {report['timestamp']}\n")
    
    # Group by severity
    findings = report['findings']
    
    for severity in ['critical', 'high', 'medium', 'low']:
        severity_findings = [f for f in findings if f['severity'] == severity]
        
        if severity_findings:
            print(f"\n## {severity.upper()} Priority\n")
            
            for finding in severity_findings:
                print(f"- [ ] [{finding['category']}] {finding['title']}")
                print(f"      {finding['recommendation']}")
    
    print("\n## General Recommendations\n")
    for rec in report['recommendations']:
        print(f"- {rec}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 generate_todo.py <audit_report.json>")
        sys.exit(1)
    
    generate_todo(sys.argv[1])
```

Usage:
```bash
python3 super_agente_auditor.py -o audit_report.json
python3 generate_todo.py audit_report.json > TODO.md
```

## Example 7: Dashboard Visualization (HTML Report)

### Script: `generate_html_report.py`
```python
#!/usr/bin/env python3
import json
import sys
from datetime import datetime

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SUPER AGENTE Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        .stats {{ display: flex; justify-content: space-around; margin: 30px 0; }}
        .stat-box {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 5px; flex: 1; margin: 0 10px; }}
        .stat-number {{ font-size: 36px; font-weight: bold; color: #007bff; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .severity-critical {{ background: #dc3545; color: white; }}
        .severity-high {{ background: #fd7e14; color: white; }}
        .severity-medium {{ background: #ffc107; color: black; }}
        .severity-low {{ background: #28a745; color: white; }}
        .severity-info {{ background: #17a2b8; color: white; }}
        .finding {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa; }}
        .finding-title {{ font-weight: bold; font-size: 18px; margin-bottom: 10px; }}
        .finding-desc {{ margin: 10px 0; }}
        .finding-rec {{ background: #e9ecef; padding: 10px; border-radius: 3px; margin-top: 10px; }}
        .recommendations {{ background: #d4edda; padding: 20px; border-radius: 5px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 SUPER AGENTE Audit Report</h1>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Project:</strong> {project_path}</p>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{total_findings}</div>
                <div class="stat-label">Total Findings</div>
            </div>
            <div class="stat-box severity-critical">
                <div class="stat-number">{critical}</div>
                <div class="stat-label">Critical</div>
            </div>
            <div class="stat-box severity-high">
                <div class="stat-number">{high}</div>
                <div class="stat-label">High</div>
            </div>
            <div class="stat-box severity-medium">
                <div class="stat-number">{medium}</div>
                <div class="stat-label">Medium</div>
            </div>
            <div class="stat-box severity-low">
                <div class="stat-number">{low}</div>
                <div class="stat-label">Low</div>
            </div>
        </div>
        
        <h2>Findings</h2>
        {findings_html}
        
        <div class="recommendations">
            <h2>Recommendations</h2>
            {recommendations_html}
        </div>
    </div>
</body>
</html>
"""

def generate_html(audit_json):
    with open(audit_json) as f:
        report = json.load(f)
    
    stats = report['statistics']
    
    # Generate findings HTML
    findings_html = ""
    for finding in report['findings']:
        findings_html += f"""
        <div class="finding">
            <div class="finding-title">[{finding['category']}] {finding['title']}</div>
            <div class="finding-desc">{finding['description']}</div>
            {f'<div class="finding-rec"><strong>Recommendation:</strong> {finding["recommendation"]}</div>' if finding['recommendation'] else ''}
        </div>
        """
    
    # Generate recommendations HTML
    recommendations_html = "<ul>"
    for rec in report['recommendations']:
        recommendations_html += f"<li>{rec}</li>"
    recommendations_html += "</ul>"
    
    html = HTML_TEMPLATE.format(
        timestamp=report['timestamp'],
        project_path=report['project_path'],
        total_findings=stats['total_findings'],
        critical=stats['by_severity']['critical'],
        high=stats['by_severity']['high'],
        medium=stats['by_severity']['medium'],
        low=stats['by_severity']['low'],
        findings_html=findings_html,
        recommendations_html=recommendations_html
    )
    
    return html

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 generate_html_report.py <audit_report.json> <output.html>")
        sys.exit(1)
    
    html = generate_html(sys.argv[1])
    with open(sys.argv[2], 'w') as f:
        f.write(html)
    
    print(f"HTML report generated: {sys.argv[2]}")
```

Usage:
```bash
python3 super_agente_auditor.py -o audit_report.json
python3 generate_html_report.py audit_report.json audit_report.html
# Open audit_report.html in a browser
```

## Example 8: Weekly Audit Automation

### Cron Job Setup
Add to crontab (`crontab -e`):

```bash
# Run audit every Monday at 9 AM
0 9 * * 1 cd /path/to/project && python3 super_agente_auditor.py -o "audit_$(date +\%Y\%m\%d).json" && python3 send_email_report.py "audit_$(date +\%Y\%m\%d).json"
```

## Tips for Effective Usage

1. **Start with a Baseline**: Run an initial audit to establish your starting point
2. **Focus on Critical Issues**: Address critical and high severity findings first
3. **Track Progress**: Save audit reports with timestamps to track improvements
4. **Automate**: Integrate into CI/CD for continuous compliance checking
5. **Customize**: Adapt the analyzers to your specific project needs
6. **Document Exceptions**: Document legitimate exceptions to audit findings
7. **Regular Cadence**: Run audits regularly (weekly/monthly) for consistent quality

## Next Steps

- Review the [AUDITOR_DOCUMENTATION.md](AUDITOR_DOCUMENTATION.md) for detailed documentation
- Check [auditor_config.json](auditor_config.json) for configuration options
- Customize analyzers for your specific requirements
- Integrate into your development workflow

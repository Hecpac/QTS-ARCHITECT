#!/usr/bin/env python3
"""
SUPER AGENTE Auditor
A comprehensive project auditor that analyzes and audits complete project structures
before any code changes, verifying compliance with BMad Method, ICT patterns,
architecture, risk management, security, and performance standards.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SeverityLevel(Enum):
    """Severity levels for audit findings"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AuditFinding:
    """Represents a single audit finding"""
    category: str
    severity: SeverityLevel
    title: str
    description: str
    location: str = ""
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "recommendation": self.recommendation
        }


@dataclass
class AuditReport:
    """Complete audit report"""
    timestamp: str
    project_path: str
    findings: List[AuditFinding] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def add_finding(self, finding: AuditFinding):
        """Add a finding to the report"""
        self.findings.append(finding)
    
    def add_recommendation(self, recommendation: str):
        """Add a general recommendation"""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)
    
    def calculate_statistics(self):
        """Calculate statistics about findings"""
        self.statistics = {
            "total_findings": len(self.findings),
            "by_severity": {
                "critical": len([f for f in self.findings if f.severity == SeverityLevel.CRITICAL]),
                "high": len([f for f in self.findings if f.severity == SeverityLevel.HIGH]),
                "medium": len([f for f in self.findings if f.severity == SeverityLevel.MEDIUM]),
                "low": len([f for f in self.findings if f.severity == SeverityLevel.LOW]),
                "info": len([f for f in self.findings if f.severity == SeverityLevel.INFO])
            },
            "by_category": {}
        }
        
        for finding in self.findings:
            category = finding.category
            if category not in self.statistics["by_category"]:
                self.statistics["by_category"][category] = 0
            self.statistics["by_category"][category] += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        self.calculate_statistics()
        return {
            "timestamp": self.timestamp,
            "project_path": self.project_path,
            "statistics": self.statistics,
            "findings": [f.to_dict() for f in self.findings],
            "recommendations": self.recommendations
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class BMadMethodAnalyzer:
    """Analyzer for BMad Method compliance"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
    
    def analyze(self, report: AuditReport):
        """Analyze BMad Method compliance"""
        # Check for BMad Method structure and patterns
        
        # Check for modular architecture
        if not self._check_modular_structure():
            report.add_finding(AuditFinding(
                category="BMad Method",
                severity=SeverityLevel.MEDIUM,
                title="Missing modular architecture",
                description="Project should follow BMad Method's modular architecture principles",
                recommendation="Organize code into well-defined modules with clear responsibilities"
            ))
        
        # Check for documentation
        if not self._check_documentation():
            report.add_finding(AuditFinding(
                category="BMad Method",
                severity=SeverityLevel.HIGH,
                title="Insufficient documentation",
                description="BMad Method requires comprehensive documentation",
                recommendation="Add README, API documentation, and inline code comments"
            ))
        
        # Check for testing structure
        if not self._check_testing_structure():
            report.add_finding(AuditFinding(
                category="BMad Method",
                severity=SeverityLevel.HIGH,
                title="Missing testing infrastructure",
                description="BMad Method emphasizes test-driven development",
                recommendation="Implement unit tests, integration tests, and test automation"
            ))
        
        report.add_recommendation("Follow BMad Method principles for maintainable and scalable architecture")
    
    def _check_modular_structure(self) -> bool:
        """Check if project has modular structure"""
        common_module_dirs = ['src', 'lib', 'modules', 'components', 'services']
        return any((self.project_path / dir_name).exists() for dir_name in common_module_dirs)
    
    def _check_documentation(self) -> bool:
        """Check if project has documentation"""
        doc_files = ['README.md', 'DOCUMENTATION.md', 'API.md', 'docs']
        has_docs = any((self.project_path / doc).exists() for doc in doc_files)
        
        # Check if README has substantial content
        readme_path = self.project_path / 'README.md'
        if readme_path.exists():
            content = readme_path.read_text()
            if len(content) > 100:  # More than just a title
                has_docs = True
        
        return has_docs
    
    def _check_testing_structure(self) -> bool:
        """Check if project has testing structure"""
        test_dirs = ['test', 'tests', '__tests__', 'spec']
        test_files = list(self.project_path.rglob('*test*.py')) + \
                     list(self.project_path.rglob('*test*.js')) + \
                     list(self.project_path.rglob('*spec*.py'))
        
        has_test_dir = any((self.project_path / dir_name).exists() for dir_name in test_dirs)
        has_test_files = len(test_files) > 0
        
        return has_test_dir or has_test_files


class ICTPatternAnalyzer:
    """Analyzer for ICT (Information and Communication Technology) patterns"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
    
    def analyze(self, report: AuditReport):
        """Analyze ICT patterns compliance"""
        
        # Check for proper API design patterns
        if not self._check_api_patterns():
            report.add_finding(AuditFinding(
                category="ICT Patterns",
                severity=SeverityLevel.MEDIUM,
                title="API design patterns not detected",
                description="Consider implementing RESTful or GraphQL API patterns",
                recommendation="Implement standardized API patterns for better interoperability"
            ))
        
        # Check for data layer patterns
        if not self._check_data_layer():
            report.add_finding(AuditFinding(
                category="ICT Patterns",
                severity=SeverityLevel.MEDIUM,
                title="Data layer patterns missing",
                description="No clear data access or repository patterns detected",
                recommendation="Implement repository pattern or data access layer for clean architecture"
            ))
        
        # Check for communication patterns
        if not self._check_communication_patterns():
            report.add_finding(AuditFinding(
                category="ICT Patterns",
                severity=SeverityLevel.LOW,
                title="Communication patterns not clearly defined",
                description="Consider implementing message queues or event-driven patterns",
                recommendation="Use established communication patterns for distributed systems"
            ))
        
        report.add_recommendation("Adopt ICT design patterns for robust system communication")
    
    def _check_api_patterns(self) -> bool:
        """Check for API patterns"""
        api_indicators = ['api', 'routes', 'controllers', 'endpoints', 'graphql']
        return any((self.project_path / indicator).exists() or 
                   list(self.project_path.rglob(f'*{indicator}*')) 
                   for indicator in api_indicators)
    
    def _check_data_layer(self) -> bool:
        """Check for data layer patterns"""
        data_indicators = ['models', 'repositories', 'dao', 'entities', 'database']
        return any((self.project_path / indicator).exists() or 
                   list(self.project_path.rglob(f'*{indicator}*')) 
                   for indicator in data_indicators)
    
    def _check_communication_patterns(self) -> bool:
        """Check for communication patterns"""
        comm_indicators = ['events', 'messaging', 'queue', 'pubsub', 'websocket']
        return any(list(self.project_path.rglob(f'*{indicator}*')) 
                   for indicator in comm_indicators)


class ArchitectureAnalyzer:
    """Analyzer for architecture patterns and best practices"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
    
    def analyze(self, report: AuditReport):
        """Analyze architecture"""
        
        # Check for separation of concerns
        if not self._check_separation_of_concerns():
            report.add_finding(AuditFinding(
                category="Architecture",
                severity=SeverityLevel.HIGH,
                title="Poor separation of concerns",
                description="Project lacks clear separation between layers and components",
                recommendation="Implement layered architecture (presentation, business logic, data access)"
            ))
        
        # Check for configuration management
        if not self._check_configuration_management():
            report.add_finding(AuditFinding(
                category="Architecture",
                severity=SeverityLevel.MEDIUM,
                title="Missing configuration management",
                description="No clear configuration management detected",
                recommendation="Use environment variables and configuration files for settings"
            ))
        
        # Check for dependency injection patterns
        if not self._check_dependency_patterns():
            report.add_finding(AuditFinding(
                category="Architecture",
                severity=SeverityLevel.LOW,
                title="Dependency management could be improved",
                description="Consider implementing dependency injection patterns",
                recommendation="Use dependency injection for better testability and maintainability"
            ))
        
        report.add_recommendation("Follow clean architecture principles for long-term maintainability")
    
    def _check_separation_of_concerns(self) -> bool:
        """Check for separation of concerns"""
        layer_indicators = ['views', 'controllers', 'services', 'models', 'components']
        found_layers = sum(1 for indicator in layer_indicators 
                          if (self.project_path / indicator).exists())
        return found_layers >= 2
    
    def _check_configuration_management(self) -> bool:
        """Check for configuration management"""
        config_files = ['.env', 'config.json', 'config.yaml', 'settings.py', 
                       'appsettings.json', 'config']
        return any((self.project_path / config).exists() for config in config_files)
    
    def _check_dependency_patterns(self) -> bool:
        """Check for dependency management"""
        dep_files = ['requirements.txt', 'package.json', 'Pipfile', 'pom.xml', 
                    'build.gradle', 'Cargo.toml', 'go.mod']
        return any((self.project_path / dep).exists() for dep in dep_files)


class RiskManagementAnalyzer:
    """Analyzer for risk management and mitigation"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
    
    def analyze(self, report: AuditReport):
        """Analyze risk management practices"""
        
        # Check for error handling
        if not self._check_error_handling():
            report.add_finding(AuditFinding(
                category="Risk Management",
                severity=SeverityLevel.HIGH,
                title="Insufficient error handling",
                description="Error handling and logging mechanisms need improvement",
                recommendation="Implement comprehensive error handling and logging strategy"
            ))
        
        # Check for backup and recovery
        if not self._check_backup_recovery():
            report.add_finding(AuditFinding(
                category="Risk Management",
                severity=SeverityLevel.MEDIUM,
                title="Missing backup and recovery plan",
                description="No backup or disaster recovery documentation found",
                recommendation="Document backup procedures and recovery plans"
            ))
        
        # Check for monitoring
        if not self._check_monitoring():
            report.add_finding(AuditFinding(
                category="Risk Management",
                severity=SeverityLevel.MEDIUM,
                title="Monitoring infrastructure missing",
                description="No monitoring or observability tools detected",
                recommendation="Implement monitoring, logging, and alerting systems"
            ))
        
        report.add_recommendation("Implement comprehensive risk management and mitigation strategies")
    
    def _check_error_handling(self) -> bool:
        """Check for error handling"""
        # Look for logging configurations or error handling patterns
        logging_files = list(self.project_path.rglob('*log*.py')) + \
                       list(self.project_path.rglob('*log*.js')) + \
                       list(self.project_path.rglob('*error*.py'))
        return len(logging_files) > 0
    
    def _check_backup_recovery(self) -> bool:
        """Check for backup and recovery documentation"""
        recovery_docs = ['BACKUP.md', 'RECOVERY.md', 'DR.md', 'disaster-recovery']
        return any((self.project_path / doc).exists() for doc in recovery_docs)
    
    def _check_monitoring(self) -> bool:
        """Check for monitoring setup"""
        monitoring_indicators = ['prometheus', 'grafana', 'monitoring', 'metrics']
        return any(list(self.project_path.rglob(f'*{indicator}*')) 
                   for indicator in monitoring_indicators)


class SecurityAnalyzer:
    """Analyzer for security best practices"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
    
    def analyze(self, report: AuditReport):
        """Analyze security practices"""
        
        # Check for exposed secrets
        if self._check_exposed_secrets():
            report.add_finding(AuditFinding(
                category="Security",
                severity=SeverityLevel.CRITICAL,
                title="Potential secrets in code",
                description="Found potential secrets or API keys in code",
                recommendation="Remove secrets from code and use environment variables or secret management"
            ))
        
        # Check for security documentation
        if not self._check_security_documentation():
            report.add_finding(AuditFinding(
                category="Security",
                severity=SeverityLevel.HIGH,
                title="Missing security documentation",
                description="No security policy or guidelines found",
                recommendation="Create SECURITY.md with security policies and vulnerability reporting"
            ))
        
        # Check for dependency vulnerabilities
        if not self._check_dependency_security():
            report.add_finding(AuditFinding(
                category="Security",
                severity=SeverityLevel.MEDIUM,
                title="Dependency security not verified",
                description="No evidence of dependency security scanning",
                recommendation="Implement automated dependency vulnerability scanning"
            ))
        
        # Check for authentication/authorization
        if not self._check_auth_patterns():
            report.add_finding(AuditFinding(
                category="Security",
                severity=SeverityLevel.HIGH,
                title="Authentication patterns not detected",
                description="No clear authentication or authorization patterns found",
                recommendation="Implement robust authentication and authorization mechanisms"
            ))
        
        report.add_recommendation("Prioritize security at every level of the application")
    
    def _check_exposed_secrets(self) -> bool:
        """Check for exposed secrets in code"""
        secret_patterns = [
            r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        for file_path in self.project_path.rglob('*.py'):
            try:
                content = file_path.read_text()
                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return True
            except Exception:
                pass
        
        return False
    
    def _check_security_documentation(self) -> bool:
        """Check for security documentation"""
        security_docs = ['SECURITY.md', 'security.md', 'SECURITY.txt']
        return any((self.project_path / doc).exists() for doc in security_docs)
    
    def _check_dependency_security(self) -> bool:
        """Check for dependency security scanning"""
        security_files = ['.snyk', 'dependabot.yml', '.github/dependabot.yml']
        return any((self.project_path / sec).exists() for sec in security_files)
    
    def _check_auth_patterns(self) -> bool:
        """Check for authentication patterns"""
        auth_indicators = ['auth', 'authentication', 'authorization', 'login', 'jwt']
        return any(list(self.project_path.rglob(f'*{indicator}*')) 
                   for indicator in auth_indicators)


class PerformanceAnalyzer:
    """Analyzer for performance standards and optimization"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
    
    def analyze(self, report: AuditReport):
        """Analyze performance considerations"""
        
        # Check for caching strategy
        if not self._check_caching():
            report.add_finding(AuditFinding(
                category="Performance",
                severity=SeverityLevel.MEDIUM,
                title="No caching strategy detected",
                description="Consider implementing caching for better performance",
                recommendation="Implement caching layer (Redis, Memcached, or in-memory caching)"
            ))
        
        # Check for database optimization
        if not self._check_database_optimization():
            report.add_finding(AuditFinding(
                category="Performance",
                severity=SeverityLevel.MEDIUM,
                title="Database optimization not evident",
                description="Consider database indexing and query optimization",
                recommendation="Review database queries and implement proper indexing"
            ))
        
        # Check for performance testing
        if not self._check_performance_testing():
            report.add_finding(AuditFinding(
                category="Performance",
                severity=SeverityLevel.LOW,
                title="Performance testing missing",
                description="No performance or load testing infrastructure found",
                recommendation="Implement performance and load testing to identify bottlenecks"
            ))
        
        # Check for async patterns
        if not self._check_async_patterns():
            report.add_finding(AuditFinding(
                category="Performance",
                severity=SeverityLevel.LOW,
                title="Asynchronous patterns not detected",
                description="Consider using async/await for I/O operations",
                recommendation="Use asynchronous programming for better concurrency"
            ))
        
        report.add_recommendation("Optimize for performance without sacrificing code clarity")
    
    def _check_caching(self) -> bool:
        """Check for caching implementation"""
        cache_indicators = ['cache', 'redis', 'memcached']
        return any(list(self.project_path.rglob(f'*{indicator}*')) 
                   for indicator in cache_indicators)
    
    def _check_database_optimization(self) -> bool:
        """Check for database optimization"""
        db_indicators = ['migrations', 'indexes', 'schema']
        return any((self.project_path / indicator).exists() or 
                   list(self.project_path.rglob(f'*{indicator}*')) 
                   for indicator in db_indicators)
    
    def _check_performance_testing(self) -> bool:
        """Check for performance testing"""
        perf_indicators = ['benchmark', 'load-test', 'performance-test', 'k6', 'jmeter']
        return any(list(self.project_path.rglob(f'*{indicator}*')) 
                   for indicator in perf_indicators)
    
    def _check_async_patterns(self) -> bool:
        """Check for async patterns"""
        for file_path in self.project_path.rglob('*.py'):
            try:
                content = file_path.read_text()
                if 'async def' in content or 'await ' in content:
                    return True
            except Exception:
                pass
        return False


class SuperAgenteAuditor:
    """
    SUPER AGENTE Auditor - Main auditor class that orchestrates all analyzers
    """
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.report = AuditReport(
            timestamp=datetime.now().isoformat(),
            project_path=str(self.project_path)
        )
        
        # Initialize all analyzers
        self.analyzers = [
            BMadMethodAnalyzer(self.project_path),
            ICTPatternAnalyzer(self.project_path),
            ArchitectureAnalyzer(self.project_path),
            RiskManagementAnalyzer(self.project_path),
            SecurityAnalyzer(self.project_path),
            PerformanceAnalyzer(self.project_path)
        ]
    
    def run_audit(self) -> AuditReport:
        """Run complete audit of the project"""
        print(f"\n{'='*80}")
        print("SUPER AGENTE AUDITOR - Project Analysis")
        print(f"{'='*80}\n")
        print(f"Analyzing project: {self.project_path}")
        print(f"Timestamp: {self.report.timestamp}\n")
        
        # Run all analyzers
        for analyzer in self.analyzers:
            analyzer_name = analyzer.__class__.__name__
            print(f"Running {analyzer_name}...")
            analyzer.analyze(self.report)
        
        # Calculate statistics
        self.report.calculate_statistics()
        
        return self.report
    
    def print_report(self):
        """Print the audit report to console"""
        print(f"\n{'='*80}")
        print("AUDIT REPORT")
        print(f"{'='*80}\n")
        
        stats = self.report.statistics
        print(f"Total Findings: {stats['total_findings']}")
        print(f"\nFindings by Severity:")
        for severity, count in stats['by_severity'].items():
            if count > 0:
                print(f"  {severity.upper()}: {count}")
        
        print(f"\nFindings by Category:")
        for category, count in stats['by_category'].items():
            print(f"  {category}: {count}")
        
        print(f"\n{'-'*80}")
        print("DETAILED FINDINGS")
        print(f"{'-'*80}\n")
        
        # Group findings by severity
        for severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, 
                        SeverityLevel.MEDIUM, SeverityLevel.LOW, SeverityLevel.INFO]:
            severity_findings = [f for f in self.report.findings if f.severity == severity]
            
            if severity_findings:
                print(f"\n{severity.value.upper()} SEVERITY:")
                print("-" * 40)
                
                for finding in severity_findings:
                    print(f"\n[{finding.category}] {finding.title}")
                    print(f"  Description: {finding.description}")
                    if finding.location:
                        print(f"  Location: {finding.location}")
                    if finding.recommendation:
                        print(f"  Recommendation: {finding.recommendation}")
        
        print(f"\n{'-'*80}")
        print("GENERAL RECOMMENDATIONS")
        print(f"{'-'*80}\n")
        
        for idx, rec in enumerate(self.report.recommendations, 1):
            print(f"{idx}. {rec}")
        
        print(f"\n{'='*80}\n")
    
    def save_report(self, output_file: str = "audit_report.json"):
        """Save the audit report to a JSON file"""
        output_path = Path(output_file)
        output_path.write_text(self.report.to_json())
        print(f"\nReport saved to: {output_path.resolve()}")


def main():
    """Main entry point for the SUPER AGENTE Auditor"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SUPER AGENTE Auditor - Comprehensive Project Analysis Tool"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to the project to audit (default: current directory)"
    )
    parser.add_argument(
        "-o", "--output",
        default="audit_report.json",
        help="Output file for the audit report (default: audit_report.json)"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only output JSON report without console output"
    )
    
    args = parser.parse_args()
    
    # Create and run auditor
    auditor = SuperAgenteAuditor(args.path)
    auditor.run_audit()
    
    # Print report unless json-only mode
    if not args.json_only:
        auditor.print_report()
    
    # Save report to file
    auditor.save_report(args.output)
    
    # Return exit code based on critical findings
    critical_count = auditor.report.statistics['by_severity']['critical']
    return 1 if critical_count > 0 else 0


if __name__ == "__main__":
    exit(main())

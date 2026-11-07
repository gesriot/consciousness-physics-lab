"""README navigation remains clickable and points to existing repo files."""
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]


class NavigationTests(unittest.TestCase):
    def test_navigation_targets_exist(self):
        text=(ROOT/'README.md').read_text()
        section=text.split('## Навигация\n',1)[1].split('\n## ',1)[0]
        bullets=[line for line in section.splitlines() if line.startswith('- ')]
        self.assertGreaterEqual(len(bullets),10)
        for line in bullets:
            match=re.search(r'\[[^\]]+\]\(([^)]+)\)',line)
            self.assertIsNotNone(match,line)
            target=match.group(1).split('#',1)[0]
            self.assertFalse('://' in target,line)
            self.assertTrue((ROOT/target).exists(),target)

if __name__=='__main__':unittest.main()

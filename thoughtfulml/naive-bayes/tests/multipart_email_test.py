import unittest
import io
import re
from naive_bayes.email_object import EmailObject

class TestMultipartEmailObject(unittest.TestCase):
  def setUp(self):
    self.multipart_file = './tests/fixtures/multipart.eml'
    self.multipart = io.open(self.multipart_file, 'r')
    self.text = self.multipart.read()
    self.multipart.seek(0)
    self.multipart_email = EmailObject(self.multipart)

  def test_parse_concatenated_body_of_text(self):
    internal_mail = self.multipart_email.mail
    assert internal_mail.is_multipart()

    body = ''
    for part in internal_mail.walk():
      if re.match("text/plain", part.get_content_type()):
        body += part.get_payload(decode=True)
      elif re.match("text/html", part.get_content_type()):
        body += part.get_payload(decode=True) 
    self.assertEqual(self.multipart_email.body(), body)

  def test_stores_subject(self):
    subject = re.search("Subject: (.*)", self.text).group(1)
    self.assertEqual(self.multipart_email.subject(), subject)

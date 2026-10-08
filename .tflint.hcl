config {
  call_module_type = "local"
}

# Bundled ruleset; the AWS plugin is added with the first AWS resources (M2).
plugin "terraform" {
  enabled = true
  preset  = "recommended"
}

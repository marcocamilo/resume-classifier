function CodeBlock(el)
  -- Check if the code-fold attribute is set to false
  if el.attr.attributes['code-fold'] == 'false' then
    -- Create the custom div with {: #open} and the code block
    local open_tag = pandoc.RawBlock('markdown', '{: #open}')
    return {open_tag, el}
  end
end
